# Divide-and-conquer ABF. run_dc_abf() fuses C subposteriors (C a power of
# two) on a binary tree: each level fuses pairs of adjacent nodes with
# run_tuned_dc_abf_node(), which takes the weighted particles of the two
# children, chooses the coalescence time T by rejection sampling, draws the
# initial particles, and runs the SMC of src/utilitiesc.cpp. With empirical
# preconditioning, the preconditioners come from the weighted covariances of
# the children. dc_make_node_params() builds the data and the fractional prior
# of a node from those of its leaves. Used by experiments/rosenbrock.R,
# experiments/nycflights13.R and experiments/backblaze.R.

dc_node_positive_int <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < 1 ||
      x != floor(x)) {
    stop(sprintf("%s must be a positive integer.", name), call. = FALSE)
  }
  as.integer(x)
}

dc_normalize_child_weights <- function(weights, n_particles, name) {
  if (!is.numeric(weights) || length(weights) != n_particles) {
    stop(sprintf("%s must be a numeric vector with one entry per particle.", name), call. = FALSE)
  }
  if (any(!is.finite(weights))) {
    stop(sprintf("%s must contain only finite values.", name), call. = FALSE)
  }
  if (any(weights < 0)) {
    stop(sprintf("%s must be nonnegative.", name), call. = FALSE)
  }
  weight_sum <- sum(weights)
  if (!is.finite(weight_sum) || weight_sum <= 0) {
    stop(sprintf("%s must have positive sum.", name), call. = FALSE)
  }
  if (abs(weight_sum - 1) > 1e-10) {
    stop(sprintf("%s must sum to one within 1e-10.", name), call. = FALSE)
  }

  weights / weight_sum
}

dc_validate_child_node <- function(child, name, n_particles = NULL, ndim = NULL) {
  required_fields <- c("out", "normw", "leaf_ids", "n_leaves",
                       "level", "node", "diagnostics")
  missing_fields <- setdiff(required_fields, names(child))
  if (length(missing_fields) > 0L) {
    stop(sprintf("%s is missing field %s.", name, missing_fields[1L]), call. = FALSE)
  }

  if (!is.matrix(child$out) || !is.numeric(child$out) || any(dim(child$out) < 1L)) {
    stop(sprintf("%s$out must be a numeric matrix with positive dimensions.", name), call. = FALSE)
  }
  if (any(!is.finite(child$out))) {
    stop(sprintf("%s$out must contain only finite values.", name), call. = FALSE)
  }
  if (!is.null(n_particles) && nrow(child$out) != n_particles) {
    stop(sprintf("%s$out has the wrong number of particle rows.", name), call. = FALSE)
  }
  if (!is.null(ndim) && ncol(child$out) != ndim) {
    stop(sprintf("%s$out has the wrong number of dimensions.", name), call. = FALSE)
  }

  child$n_leaves <- dc_node_positive_int(child$n_leaves, sprintf("%s$n_leaves", name))
  child$level <- dc_node_positive_int(child$level, sprintf("%s$level", name))
  child$node <- dc_node_positive_int(child$node, sprintf("%s$node", name))

  if (!is.numeric(child$leaf_ids) || length(child$leaf_ids) != child$n_leaves ||
      any(!is.finite(child$leaf_ids)) || any(child$leaf_ids < 1) ||
      any(child$leaf_ids != floor(child$leaf_ids))) {
    stop(sprintf("%s$leaf_ids must be positive integers matching n_leaves.", name), call. = FALSE)
  }
  child$leaf_ids <- as.integer(child$leaf_ids)
  if (is.unsorted(child$leaf_ids, strictly = TRUE)) {
    stop(sprintf("%s$leaf_ids must be ordered and unique.", name), call. = FALSE)
  }

  child
}

validate_dc_core_limit <- function(n_cores,
                                   slurm_cpus_per_task,
                                   detected_cores) {
  n_cores <- dc_node_positive_int(n_cores, "n_cores")
  if (!is.character(slurm_cpus_per_task) || length(slurm_cpus_per_task) != 1L ||
      is.na(slurm_cpus_per_task)) {
    stop("SLURM_CPUS_PER_TASK must be a character scalar.", call. = FALSE)
  }

  if (nzchar(slurm_cpus_per_task)) {
    if (!grepl("^[[:space:]]*[0-9]+[[:space:]]*$", slurm_cpus_per_task)) {
      stop("SLURM_CPUS_PER_TASK must be a positive integer.", call. = FALSE)
    }
    slurm_limit <- suppressWarnings(as.integer(slurm_cpus_per_task))
    if (is.na(slurm_limit) || slurm_limit < 1L) {
      stop("SLURM_CPUS_PER_TASK must be a positive integer.", call. = FALSE)
    }
    if (n_cores > slurm_limit) {
      stop("n_cores exceeds SLURM_CPUS_PER_TASK.", call. = FALSE)
    }
    return(n_cores)
  }

  if (!is.numeric(detected_cores) || length(detected_cores) != 1L ||
      !is.finite(detected_cores) || detected_cores < 1) {
    if (n_cores > 1L) {
      stop("n_cores > 1 requested, but the local core count could not be detected.", call. = FALSE)
    }
    return(n_cores)
  }
  if (detected_cores != floor(detected_cores)) {
    stop("detected_cores must be a positive integer.", call. = FALSE)
  }
  detected_cores <- as.integer(detected_cores)
  if (n_cores > detected_cores) {
    stop("n_cores exceeds the detected local core count.", call. = FALSE)
  }

  n_cores
}

dc_make_node_params <- function(model_name,
                                params,
                                left_leaf_ids,
                                right_leaf_ids,
                                C) {
  if (!is.list(params)) {
    stop("params must be a list.", call. = FALSE)
  }
  if (!is.character(model_name) || length(model_name) != 1L ||
      is.na(model_name) || !nzchar(model_name)) {
    stop("model_name must be a nonempty character scalar.", call. = FALSE)
  }

  expected_length <- switch(
    model_name,
    rosen5d = 6L,
    logit_wprior_countsxypairs = 5L,
    phbackblaze_centered_grouped_halfnormal = 7L,
    NA_integer_
  )
  if (is.na(expected_length)) {
    stop(sprintf("DC parameter-combination rule has not been defined for model_name = '%s'.", model_name),
         call. = FALSE)
  }
  if (length(params) != expected_length) {
    stop(sprintf("%s params must have exactly %d fields.", model_name, expected_length),
         call. = FALSE)
  }

  C <- dc_node_positive_int(C, "C")

  validate_leaf_ids <- function(leaf_ids, name) {
    if (!is.numeric(leaf_ids) || length(leaf_ids) < 1L ||
        any(!is.finite(leaf_ids)) || any(leaf_ids != floor(leaf_ids))) {
      stop(sprintf("%s must be a nonempty integer vector.", name), call. = FALSE)
    }
    leaf_ids <- as.integer(leaf_ids)
    if (is.unsorted(leaf_ids, strictly = TRUE)) {
      stop(sprintf("%s must be strictly increasing with no duplicates.", name), call. = FALSE)
    }
    if (any(leaf_ids < 1L) || any(leaf_ids > C)) {
      stop(sprintf("%s must contain only values in seq_len(C).", name), call. = FALSE)
    }
    leaf_ids
  }

  left_leaf_ids <- validate_leaf_ids(left_leaf_ids, "left_leaf_ids")
  right_leaf_ids <- validate_leaf_ids(right_leaf_ids, "right_leaf_ids")
  if (any(left_leaf_ids %in% right_leaf_ids)) {
    stop("left_leaf_ids and right_leaf_ids must be disjoint.", call. = FALSE)
  }
  if (length(left_leaf_ids) != length(right_leaf_ids)) {
    stop("left_leaf_ids and right_leaf_ids must have equal lengths.", call. = FALSE)
  }

  # Rosen5d: each parameter except the last (mu) is an exponent, so a node with
  # k leaves per child multiplies it by k.
  if (model_name == "rosen5d") {
    n_scaled_params <- length(params) - 1L
    node_params <- params
    node_params[seq_len(n_scaled_params)] <- lapply(
      params[seq_len(n_scaled_params)],
      function(value) value * length(left_leaf_ids)
    )
    return(node_params)
  }

  if (length(params[[1L]]) != C) {
    stop("C must equal length(params[[1]]).", call. = FALSE)
  }

  # Every leaf uses the same table of distinct (y, x) rows, so a node holds
  # that table with the counts of its leaves added together.
  if (model_name == "logit_wprior_countsxypairs") {
    X_list <- params[[1L]]
    Y_list <- params[[2L]]
    count_list <- params[[3L]]
    mu0 <- params[[4L]]
    prec0 <- params[[5L]]
    if (!is.list(X_list) || !is.list(Y_list) || !is.list(count_list) ||
        length(Y_list) != C || length(count_list) != C) {
      stop("logit_wprior_countsxypairs params must be list(X_list, Y_list, count_list, mu0, prec0).",
           call. = FALSE)
    }

    leaf_ids <- c(left_leaf_ids, right_leaf_ids)
    X_table <- X_list[[leaf_ids[1L]]]
    Y_table <- Y_list[[leaf_ids[1L]]]
    if (!all(vapply(X_list[leaf_ids], identical, logical(1L), X_table)) ||
        !all(vapply(Y_list[leaf_ids], identical, logical(1L), Y_table))) {
      stop("logit_wprior_countsxypairs requires every leaf to use the same table of distinct rows.",
           call. = FALSE)
    }

    left_count <- Reduce(`+`, count_list[left_leaf_ids])
    right_count <- Reduce(`+`, count_list[right_leaf_ids])
    child_prec0 <- (length(left_leaf_ids) / C) * prec0

    return(list(
      list(X_table, X_table),
      list(Y_table, Y_table),
      list(left_count, right_count),
      mu0,
      child_prec0
    ))
  }

  start_list <- params[[1L]]
  stop_list <- params[[2L]]
  event_list <- params[[3L]]
  size_z_list <- params[[4L]]
  group_list <- params[[5L]]
  G <- params[[6L]]
  prior_fraction <- params[[7L]]

  for (field in list(start_list, stop_list, event_list, size_z_list, group_list)) {
    if (!is.list(field) || length(field) != C) {
      stop("phbackblaze params must contain five data lists with one entry per shard.",
           call. = FALSE)
    }
  }

  expected_fraction <- 1 / C
  if (!is.numeric(prior_fraction) || length(prior_fraction) != C ||
      any(!is.finite(prior_fraction)) ||
      any(abs(prior_fraction - expected_fraction) > 1e-12)) {
    stop(sprintf(
      "phbackblaze original prior_fraction must equal %.17g for every leaf; got c(%s).",
      expected_fraction,
      paste(format(prior_fraction, digits = 17), collapse = ", ")
    ), call. = FALSE)
  }

  pool_field <- function(field, leaf_ids) {
    unlist(field[leaf_ids], use.names = FALSE)
  }

  left_start <- pool_field(start_list, left_leaf_ids)
  right_start <- pool_field(start_list, right_leaf_ids)
  left_stop <- pool_field(stop_list, left_leaf_ids)
  right_stop <- pool_field(stop_list, right_leaf_ids)
  left_event <- pool_field(event_list, left_leaf_ids)
  right_event <- pool_field(event_list, right_leaf_ids)
  left_size_z <- pool_field(size_z_list, left_leaf_ids)
  right_size_z <- pool_field(size_z_list, right_leaf_ids)
  left_group <- as.integer(pool_field(group_list, left_leaf_ids))
  right_group <- as.integer(pool_field(group_list, right_leaf_ids))

  list(
    list(left_start, right_start),
    list(left_stop, right_stop),
    list(left_event, right_event),
    list(left_size_z, right_size_z),
    list(left_group, right_group),
    G,
    c(sum(prior_fraction[left_leaf_ids]), sum(prior_fraction[right_leaf_ids]))
  )
}

run_dc_abf <- function(leaf_samples,
                       model_name,
                       params,
                       n_particles,
                       n_tuning_particles = n_particles,
                       seed,
                       n_cores = 1L,
                       target_rejection_acceptance = 0.01,
                       target_one_step_ess = 0.9,
                       base_step_size = 0.01,
                       n_eval_reps = 4L,
                       resamp = TRUE,
                       resamp_thr = 0.5,
                       align_phi_on_resample = TRUE,
                       preconditioning = c("no", "diagonal", "full"),
                       preconditioning_ridge = 1e-6,
                       preconditioning_max_tries = 8L,
                       step_size_min_factor = 0.5,
                       step_size_max_factor = 1.5,
                       step_size_total_min = 1e-6,
                       verbose = TRUE,
                       step_size_criterion = "ess_ratio",
                       use_offline_grid = FALSE,
                       offline_grid_refine = 1L,
                       tune_nchunks = FALSE) {
  # leaf_samples must already target L_c(theta) * pi(theta)^(1 / C).
  # With use_offline_grid = TRUE, every fusion first runs the adaptive SMC as a
  # pilot, then reruns it from the same initial particles on the pilot's time
  # grid with each step split into offline_grid_refine equal parts.
  # With tune_nchunks = TRUE, the first step is T / n, with n chosen by the
  # bisection on the one-step ESS; with FALSE, it is chosen from the spread of
  # phi across the initial particles.
  total_start <- proc.time()[["elapsed"]]
  if (!is.logical(use_offline_grid) || length(use_offline_grid) != 1L ||
      is.na(use_offline_grid)) {
    stop("use_offline_grid must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(offline_grid_refine) || length(offline_grid_refine) != 1L ||
      !is.finite(offline_grid_refine) || offline_grid_refine < 1 ||
      offline_grid_refine != floor(offline_grid_refine)) {
    stop("offline_grid_refine must be a positive integer.", call. = FALSE)
  }
  preconditioning <- match.arg(preconditioning)

  if (!is.list(leaf_samples)) {
    stop("leaf_samples must be a list.", call. = FALSE)
  }
  if (!is.character(model_name) || length(model_name) != 1L ||
      is.na(model_name) || !nzchar(model_name)) {
    stop("model_name must be a nonempty character scalar.", call. = FALSE)
  }
  if (!is.list(params)) {
    stop("params must be a list.", call. = FALSE)
  }
  C <- length(leaf_samples)
  if (C < 2L) {
    stop("C must be at least two.", call. = FALSE)
  }
  C_power_check <- C
  while (C_power_check > 1L && C_power_check %% 2L == 0L) {
    C_power_check <- C_power_check %/% 2L
  }
  if (C_power_check != 1L) {
    stop("C must be a power of two.", call. = FALSE)
  }

  n_particles <- dc_node_positive_int(n_particles, "n_particles")
  n_tuning_particles <- dc_node_positive_int(n_tuning_particles, "n_tuning_particles")
  if (preconditioning != "no" && n_particles < 2L) {
    stop("n_particles must be at least two with empirical preconditioning.", call. = FALSE)
  }

  if (!is.matrix(leaf_samples[[1L]]) || !is.numeric(leaf_samples[[1L]]) ||
      any(dim(leaf_samples[[1L]]) < 1L)) {
    stop("leaf_samples[[1]] must be a finite numeric matrix.", call. = FALSE)
  }
  ndim <- ncol(leaf_samples[[1L]])
  for (c in seq_len(C)) {
    if (!is.matrix(leaf_samples[[c]]) || !is.numeric(leaf_samples[[c]]) ||
        nrow(leaf_samples[[c]]) != n_particles || ncol(leaf_samples[[c]]) != ndim ||
        any(!is.finite(leaf_samples[[c]]))) {
      stop(sprintf("leaf_samples[[%s]] must be a finite numeric matrix with n_particles rows and ndim columns.", c), call. = FALSE)
    }
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) || seed < 1 ||
      seed != floor(seed)) {
    stop("seed must be a positive integer.", call. = FALSE)
  }
  seed_double <- as.numeric(seed)
  C_double <- as.numeric(C)
  if (seed_double + (C_double - 2) > .Machine$integer.max) {
    stop("seed is too large for the internal node seed sequence.", call. = FALSE)
  }
  if (!is.numeric(target_rejection_acceptance) || length(target_rejection_acceptance) != 1L ||
      !is.finite(target_rejection_acceptance) || target_rejection_acceptance <= 0 ||
      target_rejection_acceptance >= 1) {
    stop("target_rejection_acceptance must lie in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(target_one_step_ess) || length(target_one_step_ess) != 1L ||
      !is.finite(target_one_step_ess) || target_one_step_ess <= 0 ||
      target_one_step_ess >= 1) {
    stop("target_one_step_ess must lie in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(base_step_size) || length(base_step_size) != 1L ||
      !is.finite(base_step_size) || base_step_size <= 0) {
    stop("base_step_size must be positive.", call. = FALSE)
  }
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L ||
      !is.finite(n_eval_reps) || n_eval_reps < 1 ||
      n_eval_reps != floor(n_eval_reps) ||
      n_eval_reps > .Machine$integer.max) {
    stop("n_eval_reps must be a positive integer.", call. = FALSE)
  }
  n_eval_reps_double <- as.numeric(n_eval_reps)
  max_node_seed <- seed_double + (C_double - 2) * (n_eval_reps_double + 2)
  max_ess_seed <- max_node_seed + n_eval_reps_double
  if (max_ess_seed > .Machine$integer.max) {
    stop("seed is too large for the ESS-tuning seed sequence.", call. = FALSE)
  }
  # One tree occupies the contiguous seed block
  # seed .. seed + (C - 1) * (n_eval_reps + 2) - 1, which is what the experiment
  # scripts stride by when they schedule replicate seeds.
  if (!is.logical(resamp) || length(resamp) != 1L || is.na(resamp)) {
    stop("resamp must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(resamp_thr) || length(resamp_thr) != 1L ||
      !is.finite(resamp_thr) || resamp_thr <= 0 || resamp_thr >= 1) {
    stop("resamp_thr must lie in (0, 1).", call. = FALSE)
  }
  if (!is.logical(align_phi_on_resample) || length(align_phi_on_resample) != 1L ||
      is.na(align_phi_on_resample)) {
    stop("align_phi_on_resample must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(preconditioning_ridge) || length(preconditioning_ridge) != 1L ||
      !is.finite(preconditioning_ridge) || preconditioning_ridge < 0) {
    stop("preconditioning_ridge must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(preconditioning_max_tries) || length(preconditioning_max_tries) != 1L ||
      !is.finite(preconditioning_max_tries) || preconditioning_max_tries < 1 ||
      preconditioning_max_tries != floor(preconditioning_max_tries)) {
    stop("preconditioning_max_tries must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(step_size_min_factor) || length(step_size_min_factor) != 1L ||
      !is.finite(step_size_min_factor) ||
      !is.numeric(step_size_max_factor) || length(step_size_max_factor) != 1L ||
      !is.finite(step_size_max_factor) ||
      step_size_min_factor <= 0 || step_size_min_factor > 1 ||
      step_size_max_factor < 1) {
    stop("step size factors must satisfy 0 < step_size_min_factor <= 1 <= step_size_max_factor.", call. = FALSE)
  }
  if (!is.numeric(step_size_total_min) || length(step_size_total_min) != 1L ||
      !is.finite(step_size_total_min) || step_size_total_min < 0) {
    stop("step_size_total_min must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.character(step_size_criterion) ||
      length(step_size_criterion) != 1L ||
      !(step_size_criterion %in% c("ess_ratio", "asymptotic"))) {
    stop("step_size_criterion must be 'ess_ratio' or 'asymptotic'.", call. = FALSE)
  }
  if (!is.logical(verbose) || length(verbose) != 1L || is.na(verbose)) {
    stop("verbose must be TRUE or FALSE.", call. = FALSE)
  }

  slurm_cpus_per_task <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  detected_cores <- if (nzchar(slurm_cpus_per_task)) NA_integer_ else parallel::detectCores()
  n_cores <- validate_dc_core_limit(
    n_cores = n_cores,
    slurm_cpus_per_task = slurm_cpus_per_task,
    detected_cores = detected_cores
  )
  seed <- as.integer(seed)
  n_eval_reps <- as.integer(n_eval_reps)
  preconditioning_max_tries <- as.integer(preconditioning_max_tries)
  # Each node owns a contiguous block of n_eval_reps + 2 seeds: node_seed, then
  # its n_eval_reps ESS-tuning seeds (see run_tuned_dc_abf_node()). The last
  # seed of the block is not used; the width is kept so that the seeds are
  # those of the runs in the paper. Sibling nodes never share a seed.
  node_seed_stride <- n_eval_reps + 2L
  node_seeds <- seed + (seq_len(C - 1L) - 1L) * node_seed_stride
  n_fusions <- C - 1L
  completed_fusions <- 0L

  if (isTRUE(verbose)) {
    cat(sprintf("D&C %s ABF: C = %s, n_particles = %s, n_cores = %s\n",
                model_name, C, n_particles, n_cores))
  }

  levels <- list(vector("list", C))
  for (c in seq_len(C)) {
    levels[[1L]][[c]] <- list(
      out = leaf_samples[[c]],
      normw = rep.int(1 / n_particles, n_particles),
      leaf_ids = as.integer(c),
      n_leaves = 1L,
      level = 1L,
      node = as.integer(c),
      diagnostics = NULL
    )
  }

  node_seed_idx <- 1L
  current_level <- levels[[1L]]
  while (length(current_level) > 1L) {
    if (length(current_level) %% 2L != 0L) {
      stop("Current D&C level has odd length.", call. = FALSE)
    }
    next_level <- vector("list", length(current_level) %/% 2L)
    for (pair_idx in seq_along(next_level)) {
      left_idx <- 2L * pair_idx - 1L
      if (isTRUE(verbose)) {
        cat(sprintf(
          "Starting fusion %d/%d: level %d, node %d, shards {%s} + {%s}\n",
          completed_fusions + 1L,
          n_fusions,
          current_level[[left_idx]]$level + 1L,
          pair_idx,
          paste(current_level[[left_idx]]$leaf_ids, collapse = ", "),
          paste(current_level[[left_idx + 1L]]$leaf_ids, collapse = ", ")
        ))
        flush.console()
      }
      next_level[[pair_idx]] <- run_tuned_dc_abf_node(
        left = current_level[[left_idx]],
        right = current_level[[left_idx + 1L]],
        model_name = model_name,
        params = params,
        C = C,
        n_tuning_particles = n_tuning_particles,
        node_seed = node_seeds[node_seed_idx],
        n_cores = n_cores,
        target_rejection_acceptance = target_rejection_acceptance,
        target_one_step_ess = target_one_step_ess,
        base_step_size = base_step_size,
        n_eval_reps = n_eval_reps,
        resamp = resamp,
        resamp_thr = resamp_thr,
        align_phi_on_resample = align_phi_on_resample,
        preconditioning = preconditioning,
        preconditioning_ridge = preconditioning_ridge,
        preconditioning_max_tries = preconditioning_max_tries,
        step_size_min_factor = step_size_min_factor,
        step_size_max_factor = step_size_max_factor,
        step_size_total_min = step_size_total_min,
        step_size_criterion = step_size_criterion,
        use_offline_grid = use_offline_grid,
        offline_grid_refine = offline_grid_refine,
        tune_nchunks = tune_nchunks
      )
      node_seed_idx <- node_seed_idx + 1L
      completed_fusions <- completed_fusions + 1L
      if (isTRUE(verbose)) {
        cat(sprintf(
          "Completed fusion %d/%d (%.1f%%): level %d, node %d, %.1f seconds\n",
          completed_fusions,
          n_fusions,
          100 * completed_fusions / n_fusions,
          next_level[[pair_idx]]$level,
          next_level[[pair_idx]]$node,
          next_level[[pair_idx]]$diagnostics$elapsed_seconds
        ))
        flush.console()
      }
    }

    levels[[length(levels) + 1L]] <- next_level
    current_level <- next_level

    level_idx <- length(levels)
    expected_n_nodes <- C %/% (2L ^ (level_idx - 1L))
    if (length(current_level) != expected_n_nodes) {
      stop("Constructed D&C level has the wrong number of nodes.", call. = FALSE)
    }
    level_leaf_ids <- integer(0)
    level_n_leaves <- vapply(current_level, function(x) x$n_leaves, integer(1))
    if (length(unique(level_n_leaves)) != 1L) {
      stop("Constructed D&C level has inconsistent n_leaves.", call. = FALSE)
    }
    for (node_idx in seq_along(current_level)) {
      node <- current_level[[node_idx]]
      if (!identical(dim(node$out), c(n_particles, ndim))) {
        stop("Constructed D&C node has the wrong particle dimensions.", call. = FALSE)
      }
      if (!is.numeric(node$normw) || length(node$normw) != n_particles ||
          any(!is.finite(node$normw)) || any(node$normw < 0) ||
          abs(sum(node$normw) - 1) > 1e-10) {
        stop("Constructed D&C node has invalid normalized weights.", call. = FALSE)
      }
      if (!identical(node$level, level_idx) || !identical(node$node, as.integer(node_idx))) {
        stop("Constructed D&C node has invalid coordinates.", call. = FALSE)
      }
      level_leaf_ids <- c(level_leaf_ids, node$leaf_ids)
    }
    if (!identical(sort(level_leaf_ids), seq_len(C)) ||
        anyDuplicated(level_leaf_ids) != 0L) {
      stop("Constructed D&C level has invalid leaf_ids.", call. = FALSE)
    }
  }

  root <- current_level[[1L]]
  if (!identical(root$n_leaves, as.integer(C)) ||
      !identical(root$leaf_ids, seq_len(C))) {
    stop("Root node does not cover all leaves in order.", call. = FALSE)
  }

  list(
    root = root,
    levels = levels,
    n_shards = C,
    n_levels = length(levels),
    n_particles = n_particles,
    ndim = ndim,
    model_name = model_name,
    n_cores = n_cores,
    seed = seed,
    elapsed_seconds = proc.time()[["elapsed"]] - total_start,
    step_size_criterion = step_size_criterion,
    use_offline_grid = use_offline_grid,
    offline_grid_refine = offline_grid_refine,
    step_size_mode = if (isTRUE(use_offline_grid)) "offline_grid" else "online"
  )
}

run_tuned_dc_abf_node <- function(left,
                                  right,
                                  model_name,
                                  params,
                                  C,
                                  n_tuning_particles = NULL,
                                  node_seed,
                                  n_cores,
                                  target_rejection_acceptance = 0.01,
                                  target_one_step_ess = 0.9,
                                  base_step_size = 0.01,
                                  n_eval_reps = 4L,
                                  resamp = TRUE,
                                  resamp_thr = 0.5,
                                  align_phi_on_resample = TRUE,
                                  preconditioning = c("no", "diagonal", "full"),
                                  preconditioning_ridge = 1e-6,
                                  preconditioning_max_tries = 8L,
                                  step_size_min_factor = 0.5,
                                  step_size_max_factor = 1.5,
                                  step_size_total_min = 0,
                                  step_size_criterion = "ess_ratio",
                                  use_offline_grid = FALSE,
                                  offline_grid_refine = 1L,
                                  tune_nchunks = FALSE) {
  node_start <- proc.time()[["elapsed"]]
  if (!is.logical(use_offline_grid) || length(use_offline_grid) != 1L ||
      is.na(use_offline_grid)) {
    stop("use_offline_grid must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(tune_nchunks) || length(tune_nchunks) != 1L || is.na(tune_nchunks)) {
    stop("tune_nchunks must be TRUE or FALSE.", call. = FALSE)
  }
  preconditioning <- match.arg(preconditioning)
  C <- dc_node_positive_int(C, "C")
  if (!is.numeric(node_seed) || length(node_seed) != 1L ||
      !is.finite(node_seed) || node_seed < 1 || node_seed != floor(node_seed)) {
    stop("node_seed must be a positive integer.", call. = FALSE)
  }
  node_seed_double <- as.numeric(node_seed)
  n_cores <- dc_node_positive_int(n_cores, "n_cores")
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L ||
      !is.finite(n_eval_reps) || n_eval_reps < 1 ||
      n_eval_reps != floor(n_eval_reps) ||
      n_eval_reps > .Machine$integer.max) {
    stop("n_eval_reps must be a positive integer.", call. = FALSE)
  }
  n_eval_reps_double <- as.numeric(n_eval_reps)
  n_eval_reps <- as.integer(n_eval_reps)
  if (node_seed_double + n_eval_reps_double > .Machine$integer.max) {
    stop("node_seed is too large for the ESS-tuning seed sequence.", call. = FALSE)
  }
  if (!is.character(step_size_criterion) ||
      length(step_size_criterion) != 1L ||
      !(step_size_criterion %in% c("ess_ratio", "asymptotic"))) {
    stop("step_size_criterion must be 'ess_ratio' or 'asymptotic'.", call. = FALSE)
  }
  node_seed <- as.integer(node_seed_double)
  # This node uses node_seed .. node_seed + n_eval_reps.
  ess_tuning_seed <- node_seed + 1L

  left <- dc_validate_child_node(left, "left")
  n_particles <- nrow(left$out)
  ndim <- ncol(left$out)
  right <- dc_validate_child_node(right, "right", n_particles = n_particles, ndim = ndim)

  if (left$n_leaves != right$n_leaves) {
    stop("Children must have the same n_leaves.", call. = FALSE)
  }
  if (left$level != right$level || right$node != left$node + 1L ||
      left$node %% 2L != 1L) {
    stop(sprintf(
      "Children are not adjacent siblings: left=(level %s, node %s), right=(level %s, node %s).",
      left$level, left$node, right$level, right$node
    ), call. = FALSE)
  }
  if (any(left$leaf_ids %in% right$leaf_ids)) {
    stop("Child leaf_ids must be disjoint before combining children.", call. = FALSE)
  }

  child_cube <- array(NA_real_, dim = c(n_particles, 2L, ndim))
  child_cube[, 1L, ] <- left$out
  child_cube[, 2L, ] <- right$out

  child_weights <- cbind(
    dc_normalize_child_weights(left$normw, n_particles, "left$normw"),
    dc_normalize_child_weights(right$normw, n_particles, "right$normw")
  )

  node_params <- dc_make_node_params(
    model_name = model_name,
    params = params,
    left_leaf_ids = left$leaf_ids,
    right_leaf_ids = right$leaf_ids,
    C = C
  )
  prior_fraction_per_child <- c(left$n_leaves / C, right$n_leaves / C)
  if (model_name == "phbackblaze_centered_grouped_halfnormal") {
    node_prior_fraction <- node_params[[7L]]
    if (!isTRUE(all.equal(prior_fraction_per_child, node_prior_fraction,
                          tolerance = 1e-12, check.attributes = FALSE))) {
      stop(sprintf("%s node prior fractions do not match descendant leaf counts.", model_name),
           call. = FALSE)
    }
  }

  subposterior_means <- matrix(NA_real_, nrow = 2L, ncol = ndim)
  for (child in seq_len(2L)) {
    theta_rows <- matrix(child_cube[, child, ], nrow = n_particles, ncol = ndim)
    child_moments <- weighted_particle_moments(
      theta_rows,
      child_weights[, child],
      compute_covariance = preconditioning != "no"
    )
    subposterior_means[child, ] <- child_moments$mean
  }

  abf_preconditioner <- if (preconditioning != "no") {
    prepare_abf_preconditioner(
      init_particles = child_cube,
      preconditioning = preconditioning,
      ridge = preconditioning_ridge,
      max_tries = preconditioning_max_tries,
      particle_weights = child_weights
    )
  } else {
    NULL
  }

  set.seed(node_seed)
  if (is.null(n_tuning_particles)) n_tuning_particles <- n_particles
  n_tuning_particles <- dc_node_positive_int(n_tuning_particles, "n_tuning_particles")
  n_rejection_proposals <- as.integer(n_particles * 10)
  max_iter_rejs <- max(n_particles, n_tuning_particles) /
    target_rejection_acceptance * 10

  phase_start <- proc.time()[["elapsed"]]
  endT_fit <- tune_rejection_end_time_cpp(
    init_particles = child_cube,
    n_samples = n_rejection_proposals,
    target_acceptance = target_rejection_acceptance,
    preconditioning = preconditioning,
    abf_preconditioner = abf_preconditioner,
    child_weights = child_weights,
    n_threads = n_cores
  )
  rejection_end_time_tuning_seconds <- proc.time()[["elapsed"]] - phase_start

  # The tuning particles are used only by the search for nchunks below, so
  # they are not drawn when tune_nchunks = FALSE.
  phase_start <- proc.time()[["elapsed"]]
  tuning_particles <- if (isTRUE(tune_nchunks)) {
    rej_samp_init_part(
      out_length_tot = n_tuning_particles,
      end_time = endT_fit$end_time,
      init_part = child_cube,
      part_untouched = 0,
      max_iter = max_iter_rejs,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner,
      child_weights = child_weights,
      n_threads = n_cores
    )
  } else {
    NULL
  }
  tuning_rejection_initialization_seconds <- proc.time()[["elapsed"]] - phase_start

  # With tune_nchunks = FALSE the search is skipped and nchunks is left NULL, so
  # that the SMC takes its first step from the spread of phi across the initial
  # particles and its own rule sets every step after that.
  phase_start <- proc.time()[["elapsed"]]
  nchunks_fit <- if (isTRUE(tune_nchunks)) {
    tune_nchunks_from_one_step_ess(
      target_ess = target_one_step_ess,
      model = model_name,
      end_time = endT_fit$end_time,
      init_particles = tuning_particles,
      params = node_params,
      initial_guess = max(1L, ceiling(endT_fit$end_time / base_step_size * sqrt(ndim))),
      base_step_size = base_step_size,
      n_eval_reps = n_eval_reps,
      eval_mc_cores = 1L,
      seed = ess_tuning_seed,
      resamp = resamp,
      resamp_thr = resamp_thr,
      align_phi_on_resample = align_phi_on_resample,
      smc_function = bayesian_fusion_smc,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner,
      n_threads = n_cores
    )
  } else {
    NULL
  }
  ess_tuning_seconds <- proc.time()[["elapsed"]] - phase_start

  phase_start <- proc.time()[["elapsed"]]
  production_particles <- rej_samp_init_part(
    out_length_tot = n_particles,
    end_time = endT_fit$end_time,
    init_part = child_cube,
    part_untouched = 0,
    max_iter = max_iter_rejs,
    preconditioning = preconditioning,
    abf_preconditioner = abf_preconditioner,
    child_weights = child_weights,
    n_threads = n_cores
  )
  production_rejection_initialization_seconds <- proc.time()[["elapsed"]] - phase_start

  # The offline grid: an adaptive pilot run from the production particles, whose
  # time grid, with each step split into offline_grid_refine equal parts, is
  # then used by the production run below.
  offline_grid <- NULL
  pilot_steps <- NULL
  pilot_smc_seconds <- NULL
  if (isTRUE(use_offline_grid)) {
    phase_start <- proc.time()[["elapsed"]]
    pilot_grid <- bayesian_fusion_smc(
      end_time = endT_fit$end_time,
      nchunks = nchunks_fit$nchunks,
      model = model_name,
      init_particles = production_particles,
      params = node_params,
      init_weight = "n",
      resamp = resamp,
      resamp_thr = resamp_thr,
      only_one_step = FALSE,
      align_phi_on_resample = align_phi_on_resample,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner,
      target_one_step_ess = target_one_step_ess,
      step_size_min_factor = step_size_min_factor,
      step_size_max_factor = step_size_max_factor,
      step_size_total_min = step_size_total_min,
      step_size_criterion = step_size_criterion,
      subposterior_means = subposterior_means,
      n_threads = n_cores,
      time_grid = NULL
    )$time_mesh
    pilot_steps <- length(pilot_grid) - 1L
    offline_grid <- c(
      unlist(lapply(seq_len(pilot_steps), function(i) {
        pilot_grid[i] + (pilot_grid[i + 1L] - pilot_grid[i]) *
          (seq_len(offline_grid_refine) - 1L) / offline_grid_refine
      })),
      pilot_grid[pilot_steps + 1L]
    )
    pilot_smc_seconds <- proc.time()[["elapsed"]] - phase_start
  }

  phase_start <- proc.time()[["elapsed"]]
  smc_out <- bayesian_fusion_smc(
    end_time = endT_fit$end_time,
    nchunks = nchunks_fit$nchunks,
    model = model_name,
    init_particles = production_particles,
    params = node_params,
    init_weight = "n",
    resamp = resamp,
    resamp_thr = resamp_thr,
    only_one_step = FALSE,
    align_phi_on_resample = align_phi_on_resample,
    preconditioning = preconditioning,
    abf_preconditioner = abf_preconditioner,
    target_one_step_ess = target_one_step_ess,
    step_size_min_factor = step_size_min_factor,
    step_size_max_factor = step_size_max_factor,
    step_size_total_min = step_size_total_min,
    step_size_criterion = step_size_criterion,
    subposterior_means = subposterior_means,
    n_threads = n_cores,
    time_grid = offline_grid
  )
  production_smc_seconds <- proc.time()[["elapsed"]] - phase_start

  parent_leaf_ids <- c(left$leaf_ids, right$leaf_ids)
  parent_level <- left$level + 1L
  parent_node <- (left$node + 1L) %/% 2L

  smc_diagnostics <- smc_out
  smc_diagnostics$out <- NULL
  smc_diagnostics$normw <- NULL
  elapsed_seconds <- proc.time()[["elapsed"]] - node_start
  phase_elapsed_seconds <- list(
    rejection_end_time_tuning = rejection_end_time_tuning_seconds,
    tuning_rejection_initialization = tuning_rejection_initialization_seconds,
    ess_tuning = ess_tuning_seconds,
    pilot_smc = pilot_smc_seconds,
    production_rejection_initialization = production_rejection_initialization_seconds,
    production_smc = production_smc_seconds
  )

  diagnostics <- list(
    seed = node_seed,
    child_leaf_ids = list(left$leaf_ids, right$leaf_ids),
    prior_fraction_per_child = prior_fraction_per_child,
    child_means = subposterior_means,
    child_weights = child_weights,
    abf_preconditioner = abf_preconditioner,
    endT_fit = endT_fit,
    nchunks_fit = nchunks_fit,
    smc = smc_diagnostics,
    phase_elapsed_seconds = phase_elapsed_seconds,
    elapsed_seconds = elapsed_seconds
  )
  diagnostics["pilot_steps"] <- list(pilot_steps)
  diagnostics["step_size_mode"] <- list(if (isTRUE(use_offline_grid)) "offline_grid" else "online")

  list(
    out = smc_out$out,
    normw = smc_out$normw,
    leaf_ids = parent_leaf_ids,
    n_leaves = as.integer(left$n_leaves + right$n_leaves),
    level = parent_level,
    node = parent_node,
    diagnostics = diagnostics
  )
}
