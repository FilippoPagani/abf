# Stan sampling of the shard subposteriors for the nycflights13 and Backblaze
# experiments. Each shard is run until the bulk and tail ESS of every
# coordinate reach a target, and the draws are then thinned to n_particles.
# Used by experiments/nycflights13.R and experiments/backblaze.R.

validate_stan_shard_inputs <- function(
    parameter_names,
    n_particles,
    chains,
    warmup,
    initial_retained_per_chain,
    max_retained_per_chain,
    min_bulk_ess,
    min_tail_ess,
    enforce_rhat_le_1_01,
    enforce_zero_divergences,
    enforce_no_treedepth_hits,
    stan_seed,
    thinning_seed,
    chain_cores,
    control) {
  is_finite_scalar <- function(value) {
    is.numeric(value) && length(value) == 1L && is.finite(value)
  }

  check_whole <- function(value, name, positive = FALSE) {
    if (!is_finite_scalar(value) || value != floor(value) ||
        (positive && value <= 0) || (!positive && value < 0)) {
      qualifier <- if (positive) "a positive" else "a nonnegative"
      stop(sprintf("%s must be %s finite whole number.", name, qualifier),
           call. = FALSE)
    }
    if (is.na(suppressWarnings(as.integer(value)))) {
      stop(sprintf("%s is outside R's integer range.", name), call. = FALSE)
    }
  }

  if (!is.character(parameter_names) || length(parameter_names) == 0L ||
      anyNA(parameter_names) || any(!nzchar(parameter_names)) ||
      anyDuplicated(parameter_names)) {
    stop(
      "parameter_names must be a nonempty character vector with no missing, empty, or duplicated names.",
      call. = FALSE
    )
  }

  check_whole(n_particles, "n_particles", positive = TRUE)
  check_whole(chains, "chains", positive = TRUE)
  check_whole(warmup, "warmup")
  check_whole(initial_retained_per_chain, "initial_retained_per_chain", positive = TRUE)
  check_whole(max_retained_per_chain, "max_retained_per_chain", positive = TRUE)

  if (!is_finite_scalar(min_bulk_ess) || min_bulk_ess <= 0) {
    stop("min_bulk_ess must be a finite positive scalar.", call. = FALSE)
  }
  if (!is_finite_scalar(min_tail_ess) || min_tail_ess <= 0) {
    stop("min_tail_ess must be a finite positive scalar.", call. = FALSE)
  }

  logical_settings <- list(
    enforce_rhat_le_1_01 = enforce_rhat_le_1_01,
    enforce_zero_divergences = enforce_zero_divergences,
    enforce_no_treedepth_hits = enforce_no_treedepth_hits
  )
  logical_valid <- vapply(logical_settings, function(value) {
    is.logical(value) && length(value) == 1L && !is.na(value)
  }, logical(1L))
  if (any(!logical_valid)) {
    invalid_name <- names(logical_settings)[which(!logical_valid)[1L]]
    stop(sprintf("%s must be one non-missing logical value.", invalid_name),
         call. = FALSE)
  }

  if (isTRUE(enforce_rhat_le_1_01) && chains < 2) {
    stop("chains must be at least two when enforce_rhat_le_1_01 is TRUE.",
         call. = FALSE)
  }
  if (initial_retained_per_chain > max_retained_per_chain) {
    stop("initial_retained_per_chain must not exceed max_retained_per_chain.",
         call. = FALSE)
  }
  if (n_particles > chains * initial_retained_per_chain) {
    stop("chains * initial_retained_per_chain must be at least n_particles.",
         call. = FALSE)
  }

  check_seed <- function(value, name) {
    if (!is_finite_scalar(value) || value != floor(value) || value < 1 ||
        value > .Machine$integer.max ||
        is.na(suppressWarnings(as.integer(value)))) {
      stop(
        sprintf("%s must be a finite scalar whole number between 1 and .Machine$integer.max.", name),
        call. = FALSE
      )
    }
  }
  check_seed(stan_seed, "stan_seed")
  check_seed(thinning_seed, "thinning_seed")

  check_whole(chain_cores, "chain_cores", positive = TRUE)
  if (!(chain_cores == 1 || chain_cores == chains)) {
    stop("chain_cores must equal either 1 or chains.", call. = FALSE)
  }

  if (!is.list(control) || is.null(control$max_treedepth)) {
    stop("control must supply max_treedepth.", call. = FALSE)
  }
  check_whole(control$max_treedepth, "control$max_treedepth", positive = TRUE)

  invisible(NULL)
}

thin_stan_shard_draws <- function(
    draws_array,
    n_particles,
    seed) {
  if (!is.array(draws_array) || length(dim(draws_array)) != 3L ||
      !is.numeric(draws_array)) {
    stop("draws_array must be a numeric three-dimensional array.", call. = FALSE)
  }

  dimensions <- dim(draws_array)
  n_iterations <- dimensions[1L]
  n_chains <- dimensions[2L]
  n_variables <- dimensions[3L]
  if (any(dimensions < 1L)) {
    stop("draws_array must have positive iteration, chain, and variable dimensions.",
         call. = FALSE)
  }

  variable_names <- dimnames(draws_array)[[3L]]
  if (is.null(variable_names) || length(variable_names) != n_variables ||
      anyNA(variable_names) || any(!nzchar(variable_names)) ||
      anyDuplicated(variable_names)) {
    stop("draws_array must have unique, nonempty variable names.", call. = FALSE)
  }

  if (!is.numeric(n_particles) || length(n_particles) != 1L ||
      !is.finite(n_particles) || n_particles != floor(n_particles) ||
      n_particles <= 0 ||
      is.na(suppressWarnings(as.integer(n_particles)))) {
    stop("n_particles must be a positive finite whole number.", call. = FALSE)
  }
  n_particles <- as.integer(n_particles)

  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
      seed != floor(seed) || seed < 1 || seed > .Machine$integer.max ||
      is.na(suppressWarnings(as.integer(seed)))) {
    stop(
      "seed must be a finite scalar whole number between 1 and .Machine$integer.max.",
      call. = FALSE
    )
  }
  seed <- as.integer(seed)

  n_per_chain <- rep(n_particles %/% n_chains, n_chains)
  remainder <- n_particles %% n_chains
  if (remainder > 0L) {
    first_chains <- seq_len(remainder)
    n_per_chain[first_chains] <- n_per_chain[first_chains] + 1L
  }
  if (any(n_per_chain > n_iterations)) {
    stop(
      "draws_array has too few iterations for the requested per-chain allocation.",
      call. = FALSE
    )
  }

  had_random_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_random_seed) {
    saved_random_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_random_seed) {
      assign(".Random.seed", saved_random_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)

  selected_draws <- vector("list", n_chains)
  selected_locations <- vector("list", n_chains)
  for (chain in seq_len(n_chains)) {
    n_chain <- n_per_chain[chain]
    if (n_chain == 0L) {
      next
    }

    boundaries <- floor(seq(0, n_iterations, length.out = n_chain + 1L))
    lower_boundaries <- boundaries[-length(boundaries)] + 1L
    upper_boundaries <- boundaries[-1L]
    if (any(lower_boundaries > upper_boundaries)) {
      stop("draws_array contains an empty thinning block.", call. = FALSE)
    }
    selected_iterations <- vapply(
      seq_len(n_chain),
      function(block) {
        as.integer(sample.int(
          upper_boundaries[block] - lower_boundaries[block] + 1L,
          1L
        ) + lower_boundaries[block] - 1L)
      },
      integer(1L)
    )
    selected_draws[[chain]] <- matrix(
      as.numeric(draws_array[selected_iterations, chain, , drop = FALSE]),
      nrow = n_chain,
      ncol = n_variables,
      dimnames = list(NULL, variable_names)
    )
    selected_locations[[chain]] <- data.frame(
      chain = rep.int(as.integer(chain), n_chain),
      iteration = as.integer(selected_iterations)
    )
  }

  selected_draw_matrix <- do.call(rbind, selected_draws[n_per_chain > 0L])
  selection_table <- do.call(rbind, selected_locations[n_per_chain > 0L])
  permutation <- sample.int(n_particles, n_particles, replace = FALSE)
  selected_draw_matrix <- selected_draw_matrix[permutation, , drop = FALSE]
  selection_table <- selection_table[permutation, , drop = FALSE]
  rownames(selection_table) <- NULL
  if (!is.matrix(selected_draw_matrix) ||
      !identical(dim(selected_draw_matrix), c(n_particles, n_variables)) ||
      !identical(colnames(selected_draw_matrix), variable_names) ||
      any(!is.finite(selected_draw_matrix))) {
    stop("Selected shard draws must be finite and have the expected shape.",
         call. = FALSE)
  }
  if (!identical(names(selection_table), c("chain", "iteration")) ||
      !is.integer(selection_table$chain) ||
      !is.integer(selection_table$iteration)) {
    stop("The thinning selection table has an invalid schema.", call. = FALSE)
  }

  list(
    draws = selected_draw_matrix,
    selection = selection_table,
    n_per_chain = as.integer(n_per_chain)
  )
}

summarise_stan_shard_fit <- function(
    fit,
    parameter_names,
    max_treedepth) {
  if (!is.character(parameter_names) || length(parameter_names) == 0L ||
      anyNA(parameter_names) || any(!nzchar(parameter_names)) ||
      anyDuplicated(parameter_names)) {
    stop("parameter_names must be a nonempty character vector with unique names.",
         call. = FALSE)
  }
  if (!is.numeric(max_treedepth) || length(max_treedepth) != 1L ||
      !is.finite(max_treedepth) || max_treedepth != floor(max_treedepth) ||
      max_treedepth <= 0 ||
      is.na(suppressWarnings(as.integer(max_treedepth)))) {
    stop("max_treedepth must be a finite positive whole number.", call. = FALSE)
  }

  draws_array <- posterior::as_draws_array(
    as.array(fit, pars = parameter_names)
  )
  draw_names <- dimnames(draws_array)[[3L]]
  if (!identical(draw_names, parameter_names)) {
    stop("Stan draw variables do not match parameter_names in order.",
         call. = FALSE)
  }

  parameter_diagnostics <- posterior::summarise_draws(
    draws_array,
    ess_bulk = posterior::ess_bulk,
    ess_tail = posterior::ess_tail,
    rhat = posterior::rhat
  )
  expected_columns <- c("variable", "ess_bulk", "ess_tail", "rhat")
  if (!identical(names(parameter_diagnostics), expected_columns)) {
    stop("Stan parameter diagnostics have an unexpected column schema.",
         call. = FALSE)
  }
  if (!identical(as.character(parameter_diagnostics$variable), parameter_names)) {
    stop("Stan parameter diagnostics do not match parameter_names in order.",
         call. = FALSE)
  }

  sampler_params <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
  if (!is.list(sampler_params) || length(sampler_params) == 0L ||
      !all(vapply(sampler_params, is.matrix, logical(1L)))) {
    stop("Stan sampler diagnostics must be one matrix per chain.", call. = FALSE)
  }
  sampler_columns_valid <- vapply(sampler_params, function(chain_params) {
    all(c("divergent__", "treedepth__") %in% colnames(chain_params))
  }, logical(1L))
  if (any(!sampler_columns_valid)) {
    stop("Stan sampler output is missing divergent__ or treedepth__.",
         call. = FALSE)
  }

  divergences <- as.integer(sum(vapply(
    sampler_params,
    function(chain_params) sum(chain_params[, "divergent__"]),
    numeric(1L)
  )))
  treedepth_hits <- as.integer(sum(vapply(
    sampler_params,
    function(chain_params) sum(chain_params[, "treedepth__"] >= max_treedepth),
    numeric(1L)
  )))

  list(
    draws_array = draws_array,
    parameter_diagnostics = parameter_diagnostics,
    sampler_diagnostics = sampler_params,
    divergences = divergences,
    treedepth_hits = treedepth_hits
  )
}

sample_stan_shard_until_ess <- function(
    compiled_model,
    stan_data,
    parameter_names,
    n_particles,
    chains,
    warmup,
    initial_retained_per_chain,
    max_retained_per_chain,
    min_bulk_ess,
    min_tail_ess,
    enforce_rhat_le_1_01,
    enforce_zero_divergences,
    enforce_no_treedepth_hits,
    stan_seed,
    thinning_seed,
    chain_cores,
    control) {
  validate_stan_shard_inputs(
    parameter_names = parameter_names,
    n_particles = n_particles,
    chains = chains,
    warmup = warmup,
    initial_retained_per_chain = initial_retained_per_chain,
    max_retained_per_chain = max_retained_per_chain,
    min_bulk_ess = min_bulk_ess,
    min_tail_ess = min_tail_ess,
    enforce_rhat_le_1_01 = enforce_rhat_le_1_01,
    enforce_zero_divergences = enforce_zero_divergences,
    enforce_no_treedepth_hits = enforce_no_treedepth_hits,
    stan_seed = stan_seed,
    thinning_seed = thinning_seed,
    chain_cores = chain_cores,
    control = control
  )
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("The posterior package is required for Stan shard diagnostics.",
         call. = FALSE)
  }

  worst_diagnostic <- function(values, names, direction) {
    nonfinite <- which(!is.finite(values))
    index <- if (length(nonfinite) > 0L) {
      nonfinite[1L]
    } else if (direction == "min") {
      which.min(values)
    } else {
      which.max(values)
    }
    list(value = values[index], parameter = names[index])
  }
  format_diagnostic <- function(diagnostic) {
    sprintf("%s (%s)", format(diagnostic$value), diagnostic$parameter)
  }

  retained_per_chain <- initial_retained_per_chain
  attempt <- 1L
  attempt_history <- list()
  repeat {
    attempt_warmup <- warmup
    attempt_init <- "random"
    attempt_control <- control
    if (attempt > 1L) {
      # Resume every chain from its own last draw. Re-running the same seed from
      # scratch would only reproduce the draws that already failed, so the new
      # initial values are what let a retry sample a fresh window of the chain.
      # Adaptation restarts inside the typical set, so a short warmup with a
      # small initial buffer suffices to re-estimate the step size and metric.
      attempt_warmup <- max(1L, warmup %/% 2L)
      attempt_init <- rstan::get_inits(
        fit,
        iter = length(fit@sim$samples[[1L]][[1L]])
      )
      attempt_control$adapt_init_buffer <- 25L
    }

    fit <- rstan::sampling(
      compiled_model,
      data = stan_data,
      chains = chains,
      iter = attempt_warmup + retained_per_chain,
      warmup = attempt_warmup,
      init = attempt_init,
      seed = stan_seed,
      cores = chain_cores,
      show_messages = file.path(
        tempdir(),
        sprintf("rstan-shard-%d-%d.log", Sys.getpid(), stan_seed)
      ),
      control = attempt_control
    )
    diagnostics <- summarise_stan_shard_fit(
      fit,
      parameter_names,
      control$max_treedepth
    )

    bulk_ess_pass <- all(
      is.finite(diagnostics$parameter_diagnostics$ess_bulk) &
        diagnostics$parameter_diagnostics$ess_bulk >= min_bulk_ess
    )
    tail_ess_pass <- all(
      is.finite(diagnostics$parameter_diagnostics$ess_tail) &
        diagnostics$parameter_diagnostics$ess_tail >= min_tail_ess
    )
    rhat_pass <- all(
      is.finite(diagnostics$parameter_diagnostics$rhat) &
        diagnostics$parameter_diagnostics$rhat <= 1.01
    )
    divergence_pass <- diagnostics$divergences == 0L
    treedepth_pass <- diagnostics$treedepth_hits == 0L
    overall_pass <- bulk_ess_pass &&
      tail_ess_pass &&
      (!enforce_rhat_le_1_01 || rhat_pass) &&
      (!enforce_zero_divergences || divergence_pass) &&
      (!enforce_no_treedepth_hits || treedepth_pass)

    worst_bulk_ess <- worst_diagnostic(
      diagnostics$parameter_diagnostics$ess_bulk,
      parameter_names,
      "min"
    )
    worst_tail_ess <- worst_diagnostic(
      diagnostics$parameter_diagnostics$ess_tail,
      parameter_names,
      "min"
    )
    worst_rhat <- worst_diagnostic(
      diagnostics$parameter_diagnostics$rhat,
      parameter_names,
      "max"
    )
    attempt_history[[attempt]] <- data.frame(
      attempt = attempt,
      warmup = attempt_warmup,
      retained_per_chain = retained_per_chain,
      total_retained_draws = chains * retained_per_chain,
      min_bulk_ess = worst_bulk_ess$value,
      min_bulk_ess_parameter = worst_bulk_ess$parameter,
      min_tail_ess = worst_tail_ess$value,
      min_tail_ess_parameter = worst_tail_ess$parameter,
      max_rhat = worst_rhat$value,
      max_rhat_parameter = worst_rhat$parameter,
      divergences = diagnostics$divergences,
      treedepth_hits = diagnostics$treedepth_hits,
      bulk_ess_pass = bulk_ess_pass,
      tail_ess_pass = tail_ess_pass,
      rhat_pass = rhat_pass,
      divergence_pass = divergence_pass,
      treedepth_pass = treedepth_pass,
      overall_pass = overall_pass,
      stringsAsFactors = FALSE
    )

    if (overall_pass) {
      break
    }
    if (retained_per_chain == max_retained_per_chain) {
      failed_conditions <- c(
        if (!bulk_ess_pass) "bulk ESS" else character(),
        if (!tail_ess_pass) "tail ESS" else character(),
        if (enforce_rhat_le_1_01 && !rhat_pass) "Rhat" else character(),
        if (enforce_zero_divergences && !divergence_pass) {
          "zero divergences"
        } else {
          character()
        },
        if (enforce_no_treedepth_hits && !treedepth_pass) {
          "no treedepth hits"
        } else {
          character()
        }
      )
      stop(
        paste0(
          "Stan shard failed at the maximum retained length. ",
          "Worst bulk ESS: ", format_diagnostic(worst_bulk_ess), "; ",
          "worst tail ESS: ", format_diagnostic(worst_tail_ess), "; ",
          "maximum Rhat: ", format_diagnostic(worst_rhat), "; ",
          "divergences: ", diagnostics$divergences, "; ",
          "treedepth hits: ", diagnostics$treedepth_hits, ". ",
          "Enabled conditions failed: ", paste(failed_conditions, collapse = ", "),
          "."
        ),
        call. = FALSE
      )
    }
    # ESS grows roughly linearly in retained length, so jump straight to the
    # length the observed shortfall implies rather than always doubling: shards
    # otherwise escalate two or three times to reach a length the first attempt
    # already predicted. Never grow by less than a doubling, so failures that
    # more draws cannot fix still terminate at max_retained_per_chain.
    ess_ratio <- c(min_bulk_ess / worst_bulk_ess$value,
                   min_tail_ess / worst_tail_ess$value)
    ess_ratio <- max(1, ess_ratio[is.finite(ess_ratio)])
    retained_per_chain <- as.integer(min(
      max(2 * retained_per_chain, ceiling(1.1 * ess_ratio * retained_per_chain)),
      max_retained_per_chain
    ))
    attempt <- attempt + 1L
  }

  thinned <- thin_stan_shard_draws(
    diagnostics$draws_array,
    n_particles,
    thinning_seed
  )
  list(
    draws = thinned$draws,
    full_draws = diagnostics$draws_array,
    selection = thinned$selection,
    n_per_chain = thinned$n_per_chain,
    parameter_diagnostics = diagnostics$parameter_diagnostics,
    sampler_diagnostics = diagnostics$sampler_diagnostics,
    divergences = diagnostics$divergences,
    treedepth_hits = diagnostics$treedepth_hits,
    attempt_history = do.call(rbind, attempt_history),
    retained_per_chain = retained_per_chain,
    chains = chains,
    chain_cores = chain_cores,
    stan_seed = stan_seed,
    thinning_seed = thinning_seed
  )
}

# Samples every shard with sample_stan_shard_until_ess(), n_cores shards at a
# time, each in its own R process, and stops with the messages of the shards
# that failed. stan_seeds and thinning_seeds hold one seed per shard; the other
# arguments in ... are passed to sample_stan_shard_until_ess() unchanged.
sample_stan_shards <- function(compiled_model,
                               shard_stan_data,
                               stan_seeds,
                               thinning_seeds,
                               n_cores,
                               ...) {
  # Evaluate the arguments here, so that the worker processes receive their
  # values with sample_one_shard() and not unevaluated promises.
  force(compiled_model)
  force(shard_stan_data)
  force(stan_seeds)
  force(thinning_seeds)
  shard_args <- list(...)
  n_shards <- length(shard_stan_data)
  sample_one_shard <- function(shard) {
    cat(sprintf("Sampling shard %d of %d.\n", shard, n_shards))
    tryCatch(
      do.call(sample_stan_shard_until_ess, c(
        list(
          compiled_model = compiled_model,
          stan_data = shard_stan_data[[shard]],
          stan_seed = stan_seeds[shard],
          thinning_seed = thinning_seeds[shard]
        ),
        shard_args
      )),
      error = function(error) {
        structure(sprintf("Shard %d: %s", shard, conditionMessage(error)), class = "try-error")
      }
    )
  }

  if (n_cores == 1L) {
    results <- lapply(seq_len(n_shards), sample_one_shard)
  } else {
    cluster <- parallel::makeCluster(n_cores, outfile = "", setup_strategy = "sequential")
    on.exit(parallel::stopCluster(cluster))
    worker_library_paths <- .libPaths()
    worker_dir <- getwd()
    parallel::clusterExport(cluster, c("worker_library_paths", "worker_dir"), envir = environment())
    parallel::clusterEvalQ(cluster, {
      .libPaths(worker_library_paths)
      setwd(worker_dir)
      suppressPackageStartupMessages(library(rstan))
      source("R/stan_shard_sampling.R")
      NULL
    })
    results <- parallel::parLapplyLB(cluster, seq_len(n_shards), sample_one_shard)
  }

  failed <- vapply(results, inherits, logical(1L), "try-error")
  if (any(failed)) {
    stop(paste(unlist(results[failed]), collapse = "\n"), call. = FALSE)
  }
  results
}
