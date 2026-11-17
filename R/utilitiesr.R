# R helpers used by the runners and the experiments:
#  - seeds and argument checks;
#  - the empirical preconditioner (weighted moments, regularised Cholesky);
#  - exact i.i.d. draws from the 5d Rosenbrock target;
#  - the bisection on the number of steps n for a target one-step ESS;
#  - 2d contour plots of a sample;
#  - the simulated logistic regression: data, Random-Walk Metropolis shard
#    draws and reference sample;
#  - the table of distinct (y, x) rows of the nycflights13 shards;
#  - the integrated absolute distance (IAD) and consensus Monte Carlo (CMC).

#############################
#### Utility functions   ####
#############################

make_session_seed <- function(seed = NULL, upper = 2^15L) {
  if (is.null(seed)) {
    seed <- sample.int(as.integer(upper), 1L)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) || seed < 1) {
    stop("seed must be a positive integer.", call. = FALSE)
  }
  seed <- as.integer(seed)
  cat(sprintf("Seed for session: %s\n", seed))
  set.seed(seed)
  seed
}


as_positive_int <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x < 1) {
    stop(sprintf("%s must be a positive integer.", name), call. = FALSE)
  }
  as.integer(x)
}

validate_particle_cube <- function(x, nshards = NULL, ndim = NULL, name = "init_particles") {
  dims <- dim(x)
  if (!is.array(x) || length(dims) != 3L || any(dims < 1L)) {
    stop(sprintf("%s must be a 3D array with positive dimensions.", name), call. = FALSE)
  }
  if (!is.null(nshards) && dims[2] != nshards) {
    stop(sprintf("%s has wrong number of shard columns.", name), call. = FALSE)
  }
  if (!is.null(ndim) && dims[3] != ndim) {
    stop(sprintf("%s has wrong number of dimensions.", name), call. = FALSE)
  }
  dims
}

regularize_abf_covariance <- function(q_cov, ridge = 1e-6, max_tries = 8L) {
  if (!is.matrix(q_cov) || !is.numeric(q_cov) || nrow(q_cov) != ncol(q_cov)) {
    stop("q_cov must be a numeric square matrix.", call. = FALSE)
  }
  if (any(!is.finite(q_cov))) {
    stop("q_cov must contain only finite values.", call. = FALSE)
  }
  if (!is.numeric(ridge) || length(ridge) != 1L || !is.finite(ridge) || ridge < 0) {
    stop("ridge must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(max_tries) || length(max_tries) != 1L || !is.finite(max_tries) ||
      max_tries < 1 || max_tries != floor(max_tries)) {
    stop("max_tries must be a positive integer.", call. = FALSE)
  }

  ndim <- nrow(q_cov)
  max_tries <- as.integer(max_tries)
  q_cov <- 0.5 * (q_cov + t(q_cov))
  positive_diag <- diag(q_cov)[diag(q_cov) > 0]
  if (length(positive_diag) < 1L) {
    stop("q_cov must have at least one positive diagonal entry.", call. = FALSE)
  }

  scale_c <- mean(positive_diag)
  ridge_now <- ridge
  identity <- diag(ndim)
  for (attempt in seq_len(max_tries)) {
    ridge_added <- ridge_now * scale_c
    q_cov_regularized <- q_cov + ridge_added * identity
    chol_upper <- tryCatch(chol(q_cov_regularized), error = function(e) NULL)
    if (!is.null(chol_upper) && all(is.finite(chol_upper))) {
      return(list(
        q_cov_regularized = q_cov_regularized,
        q_chol = t(chol_upper),
        q_inv = chol2inv(chol_upper),
        ridge_added = ridge_added
      ))
    }
    if (attempt < max_tries) {
      ridge_now <- if (ridge_now > 0) ridge_now * 10 else .Machine$double.eps
    }
  }

  stop("Failed to regularize q_cov with a Cholesky factorization.", call. = FALSE)
}

weighted_particle_moments <- function(theta_rows,
                                      weights,
                                      compute_covariance = TRUE) {
  if (!is.matrix(theta_rows) || !is.numeric(theta_rows) || any(dim(theta_rows) < 1L)) {
    stop("theta_rows must be a numeric matrix with positive dimensions.", call. = FALSE)
  }
  if (any(!is.finite(theta_rows))) {
    stop("theta_rows must contain only finite values.", call. = FALSE)
  }
  if (!is.numeric(weights) || length(weights) != nrow(theta_rows)) {
    stop("weights must be a numeric vector with one entry per particle.", call. = FALSE)
  }
  if (any(!is.finite(weights))) {
    stop("weights must contain only finite values.", call. = FALSE)
  }
  if (any(weights < 0)) {
    stop("weights must be nonnegative.", call. = FALSE)
  }
  weight_sum <- sum(weights)
  if (!is.finite(weight_sum) || weight_sum <= 0) {
    stop("weights must have positive sum.", call. = FALSE)
  }
  if (abs(weight_sum - 1) > 1e-10) {
    stop("weights must sum to one within 1e-10.", call. = FALSE)
  }
  if (!is.logical(compute_covariance) || length(compute_covariance) != 1L ||
      is.na(compute_covariance)) {
    stop("compute_covariance must be TRUE or FALSE.", call. = FALSE)
  }

  weights <- weights / weight_sum
  mean <- colSums(theta_rows * weights)
  if (!compute_covariance) {
    return(list(mean = mean, covariance = NULL))
  }

  denominator <- 1 - sum(weights^2)
  if (!is.finite(denominator) || denominator <= 0) {
    stop("weights cannot define a positive covariance denominator.", call. = FALSE)
  }
  centred <- sweep(theta_rows, 2L, mean, "-")
  covariance <- crossprod(centred, centred * weights) / denominator
  covariance <- 0.5 * (covariance + t(covariance))

  list(mean = mean, covariance = covariance)
}

prepare_abf_preconditioner <- function(init_particles,
                                       preconditioning = c("diagonal", "full"),
                                       ridge = 1e-6,
                                       max_tries = 8L,
                                       particle_weights = NULL) {
  preconditioning <- match.arg(preconditioning)
  dims <- dim(init_particles)
  if (!is.array(init_particles) || length(dims) != 3L || dims[1L] < 2L ||
      dims[2L] < 1L || dims[3L] < 1L || !is.numeric(init_particles)) {
    stop("init_particles must be a numeric 3D array with at least two rows.", call. = FALSE)
  }
  if (any(!is.finite(init_particles))) {
    stop("init_particles must contain only finite values.", call. = FALSE)
  }
  if (!is.numeric(ridge) || length(ridge) != 1L || !is.finite(ridge) || ridge < 0) {
    stop("ridge must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(max_tries) || length(max_tries) != 1L || !is.finite(max_tries) ||
      max_tries < 1 || max_tries != floor(max_tries)) {
    stop("max_tries must be a positive integer.", call. = FALSE)
  }

  npart <- dims[1L]
  nshards <- dims[2L]
  ndim <- dims[3L]
  max_tries <- as.integer(max_tries)
  if (!is.null(particle_weights)) {
    if (!is.matrix(particle_weights) || !is.numeric(particle_weights) ||
        any(dim(particle_weights) != c(npart, nshards))) {
      stop("particle_weights must be a numeric matrix with one column per shard.", call. = FALSE)
    }
  }

  if (preconditioning == "diagonal") {
    q_diag <- matrix(NA_real_, nrow = nshards, ncol = ndim)
    regularization_used <- numeric(nshards)

    for (shard in seq_len(nshards)) {
      theta_rows <- matrix(init_particles[, shard, ], nrow = npart, ncol = ndim)
      raw_cov <- if (is.null(particle_weights)) {
        stats::cov(theta_rows)
      } else {
        weighted_particle_moments(theta_rows, particle_weights[, shard])$covariance
      }
      if (any(!is.finite(raw_cov))) {
        stop("Empirical covariance entries must be finite.", call. = FALSE)
      }
      raw_diag <- diag(raw_cov)
      positive_diag <- raw_diag[raw_diag > 0]
      if (length(positive_diag) < 1L) {
        stop("Each shard must have at least one positive empirical marginal variance.", call. = FALSE)
      }

      scale_c <- mean(positive_diag)
      floor_c <- max(ridge * scale_c, .Machine$double.eps * scale_c)
      q_diag[shard, ] <- pmax(raw_diag, floor_c)
      regularization_used[shard] <- floor_c
    }

    q_inv_diag <- 1 / q_diag
    q_star_diag <- 1 / colSums(q_inv_diag)
    return(list(
      mode = "diagonal",
      nshards = as.integer(nshards),
      ndim = as.integer(ndim),
      q_diag = q_diag,
      q_inv_diag = q_inv_diag,
      q_sqrt_diag = sqrt(q_diag),
      q_star_diag = q_star_diag,
      q_star_sqrt_diag = sqrt(q_star_diag),
      regularization_used = regularization_used
    ))
  }

  q_cov <- vector("list", nshards)
  q_inv <- vector("list", nshards)
  q_chol <- vector("list", nshards)
  regularization_used <- numeric(nshards)

  for (shard in seq_len(nshards)) {
    theta_rows <- matrix(init_particles[, shard, ], nrow = npart, ncol = ndim)
    raw_cov <- if (is.null(particle_weights)) {
      stats::cov(theta_rows)
    } else {
      weighted_particle_moments(theta_rows, particle_weights[, shard])$covariance
    }
    regularized <- regularize_abf_covariance(raw_cov,
                                             ridge = ridge,
                                             max_tries = max_tries)
    q_cov[[shard]] <- regularized$q_cov_regularized
    q_inv[[shard]] <- regularized$q_inv
    q_chol[[shard]] <- regularized$q_chol
    regularization_used[shard] <- regularized$ridge_added
  }

  q_star_precision <- Reduce(`+`, q_inv)
  q_star_precision <- 0.5 * (q_star_precision + t(q_star_precision))
  q_star <- chol2inv(chol(q_star_precision))
  q_star <- 0.5 * (q_star + t(q_star))
  q_star_chol_upper <- chol(q_star)
  q_star_chol <- t(q_star_chol_upper)

  list(
    mode = "full",
    nshards = as.integer(nshards),
    ndim = as.integer(ndim),
    q_cov = q_cov,
    q_inv = q_inv,
    q_chol = q_chol,
    q_star = q_star,
    q_star_chol = q_star_chol,
    regularization_used = regularization_used
  )
}

make_seed_vec <- function(seed, n, name = "seed", allow_null = TRUE) {
  n <- as_positive_int(n, "n")
  if (is.null(seed)) {
    if (allow_null) return(rep.int(NA_integer_, n))
    stop(sprintf("%s must be numeric.", name), call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) < 1L) {
    stop(sprintf("%s must be numeric.", name), call. = FALSE)
  }
  if (any(is.nan(seed)) || any(is.infinite(seed))) {
    stop(sprintf("%s must be numeric.", name), call. = FALSE)
  }
  if (any(is.na(seed))) {
    if (allow_null && all(is.na(seed))) {
      if (length(seed) == 1L) return(rep.int(NA_integer_, n))
      if (length(seed) == n) return(as.integer(seed))
    }
    stop(sprintf("%s must be numeric.", name), call. = FALSE)
  }
  if (length(seed) == 1L) return(as.integer(seed + seq_len(n) - 1L))
  if (length(seed) == n) return(as.integer(seed))
  stop(sprintf("%s must have length 1 or %s.", name, n), call. = FALSE)
}

##############################################
#### Generate initial particles functions ####
##############################################

generate_init_particles_rosen5d <- function(npart, nshards, ndim, params, safety_checks = TRUE) {
  if (isTRUE(safety_checks)) {
    npart <- as_positive_int(npart, "npart")
    nshards <- as_positive_int(nshards, "nshards")
    ndim <- as_positive_int(ndim, "ndim")
  } else {
    npart <- as.integer(npart)
    nshards <- as.integer(nshards)
    ndim <- as.integer(ndim)
  }

  if (ndim != 5L) stop("rosen5d expects ndim = 5.", call. = FALSE)

  a <- as.numeric(params[[1]])
  b12 <- as.numeric(params[[2]])
  b13 <- as.numeric(params[[3]])
  b22 <- as.numeric(params[[4]])
  b23 <- as.numeric(params[[5]])
  mu <- as.numeric(params[[6]])

  out <- array(0, dim = c(npart, nshards, ndim))
  sigma_x1 <- 1 / sqrt(2 * a)
  sigma_x12 <- 1 / sqrt(2 * b12)
  sigma_x13 <- 1 / sqrt(2 * b13)
  sigma_x22 <- 1 / sqrt(2 * b22)
  sigma_x23 <- 1 / sqrt(2 * b23)

  for (shard in seq_len(nshards)) {
    x1 <- mu + sigma_x1 * stats::rnorm(npart)
    x12 <- x1^2 + sigma_x12 * stats::rnorm(npart)
    x13 <- x12^2 + sigma_x13 * stats::rnorm(npart)
    x22 <- x1^2 + sigma_x22 * stats::rnorm(npart)
    x23 <- x22^2 + sigma_x23 * stats::rnorm(npart)
    out[, shard, 1] <- x1
    out[, shard, 2] <- x12
    out[, shard, 3] <- x13
    out[, shard, 4] <- x22
    out[, shard, 5] <- x23
  }

  out
}


############################################################
#### One-step ESS tuning for the bridge discretisation. ####
############################################################

evaluate_one_step_ess_ratio <- function(nchunks,
                                        model,
                                        end_time,
                                        init_particles,
                                        params,
                                        n_eval_reps = 10L,
                                        eval_mc_cores = 1L,
                                        seed = NULL,
                                        resamp = TRUE,
                                        resamp_thr = 0.5,
                                        align_phi_on_resample = TRUE,
                                        smc_function = bayesian_fusion_smc,
                                        preconditioning = "no",
                                        abf_preconditioner = NULL,
                                        n_threads = 1L) {
  # used when selecting nChunks. For a step_size, evaluates ESS/N for one step
  
  # safety checks
  if (!is.numeric(nchunks) || length(nchunks) != 1L || !is.finite(nchunks) || nchunks < 1) {
    stop("nchunks must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(end_time) || length(end_time) != 1L || !is.finite(end_time) || end_time <= 0) {
    stop("end_time must be a positive finite number.", call. = FALSE)
  }
  nchunks <- as.integer(nchunks)
  model <- match.arg(model, c("rosen5d", "logit", "logit_wprior_countsxypairs", "phbackblaze_centered_grouped_halfnormal"))
  dims <- dim(init_particles)
  if (!is.array(init_particles) || length(dims) != 3L || any(dims < 1L)) {
    stop("init_particles must be a 3D array with positive dimensions.", call. = FALSE)
  }
  dims <- list(npart = dims[1], nshards = dims[2], ndim = dims[3])
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L || !is.finite(n_eval_reps) || n_eval_reps < 1 || n_eval_reps != floor(n_eval_reps)) {
    stop("n_eval_reps must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(eval_mc_cores) || length(eval_mc_cores) != 1L || !is.finite(eval_mc_cores) || eval_mc_cores < 1 || eval_mc_cores != floor(eval_mc_cores)) {
    stop("eval_mc_cores must be a positive integer.", call. = FALSE)
  }
  if (!is.function(smc_function)) {
    stop("smc_function must be a function.", call. = FALSE)
  }
  preconditioning <- match.arg(preconditioning, c("no", "diagonal", "full"))
  is_approximate_abf <- identical(smc_function, bayesian_fusion_smc)
  if (preconditioning != "no" && !is_approximate_abf) {
    stop("Empirical preconditioning is supported only by bayesian_fusion_smc().", call. = FALSE)
  }
  n_eval_reps <- as.integer(n_eval_reps)
  eval_mc_cores <- min(as.integer(eval_mc_cores), n_eval_reps)
  seeds <- make_seed_vec(seed, n_eval_reps, name = "seed", allow_null = eval_mc_cores == 1L)

  evaluate_repetition <- function(i) {
    if (!is.na(seeds[i])) {
      set.seed(seeds[i])
    }

    if (is_approximate_abf) {
      out <- smc_function(
        end_time = end_time,
        nchunks = nchunks,
        model = model,
        init_particles = init_particles,
        params = params,
        init_weight = "n",
        resamp = resamp,
        resamp_thr = resamp_thr,
        only_one_step = TRUE,
        align_phi_on_resample = align_phi_on_resample,
        preconditioning = preconditioning,
        abf_preconditioner = abf_preconditioner,
        n_threads = n_threads
      )
    } else {
      out <- smc_function(
        end_time = end_time,
        nchunks = nchunks,
        model = model,
        init_particles = init_particles,
        params = params,
        init_weight = "n",
        resamp = resamp,
        resamp_thr = resamp_thr,
        only_one_step = TRUE,
        align_phi_on_resample = align_phi_on_resample
      )
    }
    essw(out$normw) / dims$npart # compute normalised ESS \in (0,1]
  }

  if (eval_mc_cores == 1L) {
    ess_vals <- vapply(seq_len(n_eval_reps), evaluate_repetition, numeric(1))
  } else {
    ess_list <- parallel::mclapply(
      seq_len(n_eval_reps),
      evaluate_repetition,
      mc.cores = eval_mc_cores,
      mc.preschedule = TRUE,
      mc.allow.recursive = FALSE,
      mc.set.seed = FALSE
    )
    failed <- vapply(ess_list, inherits, logical(1), "try-error")
    if (any(failed)) {
      stop(sprintf("Parallel ESS evaluation failed for repetition %s.", which(failed)[1]), call. = FALSE)
    }
    ess_vals <- vapply(ess_list, as.numeric, numeric(1))
  }

  list(estimate = mean(ess_vals), per_rep = ess_vals, seeds = seeds, npart = dims$npart)
}

# The cache avoids repeating the same expensive SMC call during bracketing.

make_cached_integer_evaluator <- function(eval_fn) {
  # This is used in the nchunks tuning path
  # We cache during the search for nChunks_hat because the search procedure may hit the same value
  
  # when a function is evaluated for an integer, cache the eval and re-use it
  cache <- new.env(parent = emptyenv()) # create an environment to store results
  # Because it's fast key-value lookup, mutable storage, easy to share between nested functions
  # parent = emptyenv() means it does not inherit anything

  evaluate <- function(n) {
    if (!is.numeric(n) || length(n) != 1L || !is.finite(n) || n < 1) {
      stop("n must be a positive integer.", call. = FALSE) # validate input
    }
    n <- as.integer(n) # caching is for integers only
    key <- as.character(n) # creates a string key for lookup
    if (!exists(key, envir = cache, inherits = FALSE)) { # Compute only if not already cached
      assign(key, eval_fn(n), envir = cache)
    }
    get(key, envir = cache, inherits = FALSE) # Return cached result
  }

  # evaluate: the cached wrapper you call
  # cache: the raw environment containing stored results
  list(evaluate = evaluate, cache = cache)
}


cache_to_data_frame <- function(cache_env, value_name) {
  # Turns the integer-keyed cache environment from make_cached_integer_evaluator
  # into a plain data frame you can inspect or return. Each stored object is
  # expected to be a list with an $estimate field.
  keys <- ls(envir = cache_env, all.names = TRUE)
  if (length(keys) == 0L) {
    out <- data.frame(nchunks = integer(0), estimate = numeric(0))
    names(out)[2] <- value_name
    return(out)
  }

  nchunks <- as.integer(keys)
  order_idx <- order(nchunks)
  values <- vapply(keys, function(k) {
    get(k, envir = cache_env, inherits = FALSE)$estimate
  }, numeric(1))

  out <- data.frame(nchunks = nchunks[order_idx], estimate = values[order_idx])
  names(out)[2] <- value_name
  out
}


tune_nchunks_from_one_step_ess <- function(target_ess,
                                           model,
                                           end_time,
                                           init_particles,
                                           params,
                                           initial_guess = NULL,
                                           base_step_size = 0.01,
                                           n_eval_reps = 10L,
                                           eval_mc_cores = 1L,
                                           seed = NULL,
                                           resamp = TRUE,
                                           resamp_thr = 0.5,
                                           max_expand = 25L,
                                           align_phi_on_resample = TRUE,
                                           smc_function = bayesian_fusion_smc,
                                           preconditioning = "no",
                                           abf_preconditioner = NULL,
                                           n_threads = 1L) {
  # find the smallest integer nchunks such that
  # estimated one-step ESS ratio >= target_ess.

  # safety checks and some assignments
  if (!is.numeric(target_ess) || length(target_ess) != 1L || !is.finite(target_ess) ||
      target_ess <= 0 || target_ess >= 1) {
    stop("target_ess must lie in (0, 1).", call. = FALSE)
  }
  model <- match.arg(model, c("rosen5d", "logit", "logit_wprior_countsxypairs", "phbackblaze_centered_grouped_halfnormal"))
  dims <- dim(init_particles)
  if (!is.array(init_particles) || length(dims) != 3L || any(dims < 1L)) {
    stop("init_particles must be a 3D array with positive dimensions.", call. = FALSE)
  }
  dims <- list(npart = dims[1], nshards = dims[2], ndim = dims[3])
  if (!is.numeric(base_step_size) || length(base_step_size) != 1L || !is.finite(base_step_size) || base_step_size <= 0) {
    stop("base_step_size must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L || !is.finite(n_eval_reps) || n_eval_reps < 1 || n_eval_reps != floor(n_eval_reps)) {
    stop("n_eval_reps must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(eval_mc_cores) || length(eval_mc_cores) != 1L || !is.finite(eval_mc_cores) || eval_mc_cores < 1 || eval_mc_cores != floor(eval_mc_cores)) {
    stop("eval_mc_cores must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(max_expand) || length(max_expand) != 1L || !is.finite(max_expand) || max_expand < 1) {
    stop("max_expand must be a positive integer.", call. = FALSE)
  }
  if (!is.function(smc_function)) {
    stop("smc_function must be a function.", call. = FALSE)
  }
  preconditioning <- match.arg(preconditioning, c("no", "diagonal", "full"))
  if (preconditioning != "no" && !identical(smc_function, bayesian_fusion_smc)) {
    stop("Empirical preconditioning is supported only by bayesian_fusion_smc().", call. = FALSE)
  }
  n_eval_reps <- as.integer(n_eval_reps)
  eval_mc_cores <- min(as.integer(eval_mc_cores), n_eval_reps)
  max_expand <- as.integer(max_expand)

  warm_start <- !is.null(initial_guess)
  if (!warm_start) {
    initial_guess <- max(1L, ceiling(end_time / base_step_size * sqrt(dims$ndim)))
  }
  if (!is.numeric(initial_guess) || length(initial_guess) != 1L || !is.finite(initial_guess) ||
      initial_guess < 1 || initial_guess != floor(initial_guess)) {
    stop("initial_guess must be a positive integer.", call. = FALSE)
  }
  initial_guess <- as.integer(initial_guess)

  # seed handling
  seed_vec <- make_seed_vec(seed, n_eval_reps, name = "seed", allow_null = eval_mc_cores == 1L)
  ## ##
  
  # call evaluate_one_step_ess_ratio and cache results
  cached <- make_cached_integer_evaluator(function(nchunks) {
    evaluate_one_step_ess_ratio(
      nchunks = nchunks,
      model = model,
      end_time = end_time,
      init_particles = init_particles,
      params = params,
      n_eval_reps = n_eval_reps,
      eval_mc_cores = eval_mc_cores,
      seed = seed_vec,
      resamp = resamp,
      resamp_thr = resamp_thr,
      align_phi_on_resample = align_phi_on_resample,
      smc_function = smc_function,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner,
      n_threads = n_threads
    )
  })
  ess_at <- cached$evaluate 
  # this is a function object extracted from the list returned by make_cached_integer_evaluator(...)

  expand_iter <- 0L

  if (!warm_start) {
    ess_one <- ess_at(1L)$estimate
    if (ess_one >= target_ess) {
      return(list(nchunks = 1L, target_ess = target_ess, ess_estimate = ess_one,
                  evaluations = cache_to_data_frame(cached$cache, "ess_estimate")))
    }
    lo <- 1L
    hi <- max(2L, initial_guess)
    ess_hi <- ess_at(hi)$estimate
    while (ess_hi < target_ess) {
      if (expand_iter >= max_expand) stop("Failed to find a feasible nchunks value.", call. = FALSE)
      hi <- hi * 2L
      ess_hi <- ess_at(hi)$estimate
      expand_iter <- expand_iter + 1L
    }
  } else {
    # An explicit guess is a warm start: bracket around it without first
    # evaluating a potentially expensive one-step path at nchunks = 1.
    ess_initial <- ess_at(initial_guess)$estimate

    if (ess_initial < target_ess) {
      lo <- initial_guess
      hi <- max(initial_guess + 1L, initial_guess * 2L)
      ess_hi <- ess_at(hi)$estimate
      while (ess_hi < target_ess) {
        if (expand_iter >= max_expand) stop("Failed to find a feasible nchunks value.", call. = FALSE)
        lo <- hi
        hi <- hi * 2L
        ess_hi <- ess_at(hi)$estimate
        expand_iter <- expand_iter + 1L
      }
    } else {
      hi <- initial_guess
      if (hi == 1L) {
        return(list(nchunks = 1L, target_ess = target_ess, ess_estimate = ess_initial,
                    evaluations = cache_to_data_frame(cached$cache, "ess_estimate")))
      }

      repeat {
        if (expand_iter >= max_expand) stop("Failed to find an infeasible nchunks value.", call. = FALSE)
        lo <- max(1L, hi %/% 2L)
        ess_lo <- ess_at(lo)$estimate
        expand_iter <- expand_iter + 1L
        if (ess_lo < target_ess) break
        hi <- lo
        if (hi == 1L) {
          return(list(nchunks = 1L, target_ess = target_ess, ess_estimate = ess_lo,
                      evaluations = cache_to_data_frame(cached$cache, "ess_estimate")))
        }
      }
    }
  }

  while (lo + 1L < hi) {
    mid <- lo + (hi - lo) %/% 2L
    ess_mid <- ess_at(mid)$estimate
    if (ess_mid >= target_ess) hi <- mid else lo <- mid
  }

  list(nchunks = hi,
       target_ess = target_ess,
       ess_estimate = ess_at(hi)$estimate,
       evaluations = cache_to_data_frame(cached$cache, "ess_estimate")) # useful for inspecting what the search actually did.
}


#############################
#### 2D contour plotting ####
#############################


plot_ctour_hist_2d <- function(res3, pix_size, conf_levels, create_canvas = TRUE, ...) {
  stopifnot(dim(res3)[2] == 2)
  library(IDPmisc)
  library(RColorBrewer)

  myImage <- function(x, y = NULL, pixs = 1, zmax = NULL,
                      ztransf = function(x) x,
                      colramp = IDPcolorRamp,
                      factors = c(FALSE, FALSE),
                      matrix = FALSE) {
    xy <- NaRV.omit(getXY(x, y))
    factors <- factors | sapply(xy, is.factor)
    xy <- sapply(xy, as.numeric)
    pixs <- (pixs / 10) / 2.54
    usr <- par("usr")
    bx <- if (factors[1]) {
      seq(min(xy[, 1] - 0.25), max(xy[, 1] + 0.25), length.out = 2 * diff(range(xy[, 1])) + 2)
    } else {
      seq(usr[1], usr[2], length.out = round(par("pin") / pixs)[1] + 1)
    }
    by <- if (factors[2]) {
      seq(min(xy[, 2] - 0.25), max(xy[, 2] + 0.25), length.out = 2 * diff(range(xy[, 2])) + 2)
    } else {
      seq(usr[3], usr[4], length.out = round(par("pin") / pixs)[2] + 1)
    }

    zz <- ztransf(table(cut(xy[, 1], breaks = bx), cut(xy[, 2], breaks = by)))
    zzmax <- ceiling(max(zz))
    if (is.null(zmax)) zmax <- zzmax else zmax <- ceiling(zmax)
    if (zmax < 1 || is.null(zmax)) stop("plot(x,y,...) must be called before plot_ctour_hist_2d().")
    if (zzmax > zmax) stop("zmax too small.", call. = FALSE)
    zmax <- max(zmax, 2)

    xx <- 0.5 * (bx[-1] + bx[-length(bx)])
    yy <- 0.5 * (by[-1] + by[-length(by)])
    if (matrix) invisible(list(xx, yy, zz, bx, by)) else invisible(zzmax)
  }

  conf_contour2 <- function(hist2d_obj, alpha, normalise = FALSE, add = FALSE,
                            cl = "black", lab = TRUE, Xlim, Ylim) {
    znorm <- hist2d_obj[[3]] / ifelse(normalise, sum(hist2d_obj[[3]]), 1)
    zdens <- sort(znorm, decreasing = TRUE)
    Czdens <- cumsum(zdens)
    Czdens <- Czdens / Czdens[length(Czdens)]

    for (cont.level in seq_along(alpha)) {
      tmpp <- max(which(Czdens <= alpha[cont.level]))
      if (is.infinite(tmpp)) next
      crit.val <- zdens[tmpp]
      contour(x = hist2d_obj[[1]], y = hist2d_obj[[2]], z = znorm,
              levels = crit.val,
              add = ifelse(add, TRUE, ifelse(cont.level == 1, FALSE, TRUE)),
              method = "flattest",
              labels = alpha[cont.level],
              drawlabels = lab,
              col = cl[cont.level],
              xlim = Xlim,
              ylim = Ylim)
    }
  }

  r_col <- colorRampPalette(rev(brewer.pal(11, "Spectral")))(length(conf_levels))
  if (isTRUE(create_canvas)) {
    plot(res3, type = "n", ann = FALSE, ...)
  }
  histMat <- myImage(res3, pixs = pix_size, matrix = TRUE)
  usr <- par("usr")
  conf_contour2(histMat, alpha = conf_levels, cl = r_col, lab = FALSE,
                add = TRUE, Xlim = usr[1:2], Ylim = usr[3:4])
}


######################################################################
#### Logit RW helpers for the Laplace-preconditioned tuning path. ####
######################################################################

chol_with_ridge <- function(Sigma, ridge = 1e-6, max_tries = 8L) {
  # regularise covariance Sigma by adding a small ridge alpha*I
  # keep adding until Cholesky succeeds.
  
  Sigma <- as.matrix(Sigma)
  if (!is.numeric(Sigma) || nrow(Sigma) != ncol(Sigma)) {
    stop("Sigma must be a numeric square matrix.", call. = FALSE)
  }

  d <- nrow(Sigma)
  Sigma[!is.finite(Sigma)] <- 0
  Sigma <- 0.5 * (Sigma + t(Sigma))

  ridge_now <- max(0, ridge)
  for (k in 0:max_tries) {
    Sigma_try <- Sigma + diag(ridge_now, d)
    chol_upper <- tryCatch(chol(Sigma_try), error = function(e) NULL)
    if (!is.null(chol_upper) && all(is.finite(chol_upper))) {
      return(list(
        chol = t(chol_upper),
        matrix = Sigma_try,
        ridge_used = ridge_now
      ))
    }
    ridge_now <- if (ridge_now > 0) ridge_now * 10 else .Machine$double.eps
  }

  stop("Failed to compute a Cholesky factor even after ridge regularisation.", call. = FALSE)
}


safe_glm_coef <- function(glm_fit, ndim) {
  coef <- as.numeric(stats::coef(glm_fit))
  if (length(coef) != ndim) stop("safe_glm_coef: length(coef) != ndim")
  if (any(!is.finite(coef))) stop("safe_glm_coef: any(!is.finite(coef))")
  
  coef
}


laplace_rw_metric <- function(glm_fit, ridge = 1e-6) {
  # takes a fitted logistic glm object and turns its estimated covariance 
  # into a proposal metric for the logit random-walk sampler.
  coef <- as.numeric(stats::coef(glm_fit))
  ndim <- length(coef)

  raw <- tryCatch(stats::vcov(glm_fit), error = function(e) NULL) # try to get glm cov.
  if (is.null(raw) || !is.matrix(raw) || any(dim(raw) != c(ndim, ndim))) {
    stop("laplace_rw_metric: GLM Hessian has a problem.")
  } else {
    raw <- as.matrix(raw)
  }

  if (any(!is.finite(raw))) stop("laplace_rw_metric: any(!is.finite(raw))")

  chol_info <- chol_with_ridge(raw, ridge = ridge) # this regularises raw with a ridge.
  list(
    metric = chol_info$matrix,
    chol = chol_info$chol,
    ridge_used = chol_info$ridge_used
  )
}


tune_logit_rw_scale <- function(Y,
                                X,
                                init,
                                chol_metric,
                                sigma_init,
                                target_acc = 0.35,
                                batch_size = 50L,
                                n_batches = 40L,
                                gamma0 = 0.05,
                                gamma_decay = 0.5,
                                seed = NULL) {
  # just runs the random walk to tune the RW step size.
  
  # safety checks
  if (!is.numeric(batch_size) || length(batch_size) != 1L || !is.finite(batch_size) || batch_size < 1) {
    stop("batch_size must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(n_batches) || length(n_batches) != 1L || !is.finite(n_batches) || n_batches < 1) {
    stop("n_batches must be a positive integer.", call. = FALSE)
  }
  if (!is.numeric(sigma_init) || length(sigma_init) != 1L || !is.finite(sigma_init) || sigma_init <= 0) {
    stop("sigma_init must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(target_acc) || length(target_acc) != 1L || !is.finite(target_acc) || target_acc <= 0 || target_acc >= 1) {
    stop("target_acc must lie in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(gamma0) || length(gamma0) != 1L || !is.finite(gamma0) || gamma0 <= 0) {
    stop("gamma0 must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(gamma_decay) || length(gamma_decay) != 1L || !is.finite(gamma_decay) || gamma_decay < 0) {
    stop("gamma_decay must be a nonnegative finite number.", call. = FALSE)
  }
  batch_size <- as.integer(batch_size)
  n_batches <- as.integer(n_batches)

  seed <- make_seed_vec(seed, 1L, name = "seed", allow_null = FALSE)[1]
  set.seed(seed)
  ## ##

  rwLogitTuneChol(
    Y = Y,
    X = X,
    xi = init,
    sigma_init = sigma_init,
    batch_size = batch_size,
    n_batches = n_batches,
    chol_A = chol_metric,
    target_acc = target_acc,
    gamma0 = gamma0,
    gamma_decay = gamma_decay
  )
}

#######################################
#### Logit data-generation helper. ####
#######################################

generate_logit_data_and_shards <- function(true_betas,
                                           ndata,
                                           nshards,
                                           npart,
                                           Nmctruth = 120000,
                                           target_acc_rw = 0.35,
                                           UpperRW = 0.2,
                                           n_truth_chains = 1L, # run MCtruth on a single core by default
                                           tune_batch_size = 50L, #50
                                           tune_n_batches = 50L, #40
                                           n_tune_shards = nshards,
                                           tune_seed = NULL,
                                           laplace_ridge = 1e-6,
                                           adapt_gamma0 = 0.5, #0.05
                                           adapt_gamma_decay = 0.5,
                                           compute_truth = TRUE,
                                           tune_parallel = TRUE,
                                           tune_mc_cores = NULL,
                                           chain_parallel = TRUE,
                                           chain_mc_cores = nshards,
                                           safety_checks = TRUE) {
  # safety checks
  true_betas <- as.numeric(true_betas)
  ndim <- length(true_betas)

  if (isTRUE(safety_checks)) {
    if (ndim < 1L || any(!is.finite(true_betas))) {
      stop("true_betas must contain at least one finite value.", call. = FALSE)
    }
    ndata <- as_positive_int(ndata, "ndata")
    nshards <- as_positive_int(nshards, "nshards")
    npart <- as_positive_int(npart, "npart")
    Nmctruth <- as_positive_int(Nmctruth, "Nmctruth")
    n_truth_chains <- as_positive_int(n_truth_chains, "n_truth_chains")
    tune_batch_size <- as_positive_int(tune_batch_size, "tune_batch_size")
    tune_n_batches <- as_positive_int(tune_n_batches, "tune_n_batches")
    n_tune_shards <- as_positive_int(n_tune_shards, "n_tune_shards")
  } else {
    ndata <- as.integer(ndata)
    nshards <- as.integer(nshards)
    npart <- as.integer(npart)
    Nmctruth <- as.integer(Nmctruth)
    n_truth_chains <- as.integer(n_truth_chains)
    tune_batch_size <- as.integer(tune_batch_size)
    tune_n_batches <- as.integer(tune_n_batches)
    n_tune_shards <- as.integer(n_tune_shards)
  }

  if (isTRUE(compute_truth) && Nmctruth %% n_truth_chains != 0L) {
    stop("Nmctruth must be divisible by n_truth_chains.", call. = FALSE)
  }

  if (!is.numeric(UpperRW) || length(UpperRW) != 1L || !is.finite(UpperRW) || UpperRW <= 0) {
    stop("UpperRW must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(target_acc_rw) || length(target_acc_rw) != 1L || !is.finite(target_acc_rw) || target_acc_rw <= 0 || target_acc_rw >= 1) {
    stop("target_acc_rw must lie in (0, 1).", call. = FALSE)
  }
  if (!is.numeric(laplace_ridge) || length(laplace_ridge) != 1L || !is.finite(laplace_ridge) || laplace_ridge < 0) {
    stop("laplace_ridge must be a nonnegative finite number.", call. = FALSE)
  }
  if (!is.numeric(adapt_gamma0) || length(adapt_gamma0) != 1L || !is.finite(adapt_gamma0) || adapt_gamma0 <= 0) {
    stop("adapt_gamma0 must be a positive finite number.", call. = FALSE)
  }
  if (!is.numeric(adapt_gamma_decay) || length(adapt_gamma_decay) != 1L || !is.finite(adapt_gamma_decay) || adapt_gamma_decay < 0) {
    stop("adapt_gamma_decay must be a nonnegative finite number.", call. = FALSE)
  }
  # This forces n_tune_shards to lie between 1 and nshards.
  n_tune_shards <- max(1L, min(n_tune_shards, nshards)) # this allows you to do the tuning on a fraction of the shards
  tune_seed_vec <- make_seed_vec(tune_seed, n_tune_shards, name = "tune_seed", allow_null = FALSE)
  tune_seed_base <- tune_seed_vec[1]
  ## ##

  dataX_list <- vector("list", nshards)
  dataY_list <- vector("list", nshards)
  GLM_list <- vector("list", nshards)
  GLM_coef_list <- vector("list", nshards)

  # for each shard...
  for (shard in seq_len(nshards)) {
    # ...generate logit data...
    X <- matrix(rnorm(ndata * ndim), nrow = ndata, ncol = ndim)
    X[, 1] <- 1 # intercept

    linpred <- drop(X %*% true_betas)
    Y <- rbinom(ndata, 1, plogis(linpred))

    # ...and estimate the GLM.
    GLM <- stats::glm(Y ~ . - 1, family = stats::binomial(), data = as.data.frame(X))
    GLM$coefficients <- safe_glm_coef(GLM, ndim) # regularises coefficients (e.g. stops if they're infinite)

    dataX_list[[shard]] <- X
    dataY_list[[shard]] <- Y
    GLM_list[[shard]] <- GLM
    GLM_coef_list[[shard]] <- GLM$coefficients
  }

  Params <- list(dataX_list, dataY_list)

  # create a list holding the regularised Hessians (via laplace_rw_metric()) for each shard
  shard_metric_list <- lapply(GLM_list, laplace_rw_metric, ridge = laplace_ridge)

  tune_ids <- seq_len(n_tune_shards)

  # This decides how many processes to use when tuning the shard scales
  # forces the number of parallel worker processes to lie between 1 and the number of shard-tuning tasks.
  if (is.null(tune_mc_cores)) {
    tune_mc_cores <- length(tune_ids)
  }
  tune_mc_cores <- max(1L, min(as.integer(tune_mc_cores), length(tune_ids)))

  # define shard worker for tuning RWs in parallel via mclapply
  shard_tune_worker <- function(j) {
    i <- tune_ids[j]
    tune_logit_rw_scale(
      Y = dataY_list[[i]],
      X = dataX_list[[i]],
      init = GLM_coef_list[[i]],
      chol_metric = shard_metric_list[[i]]$chol,
      sigma_init = UpperRW,
      target_acc = target_acc_rw,
      batch_size = tune_batch_size,
      n_batches = tune_n_batches,
      gamma0 = adapt_gamma0,
      gamma_decay = adapt_gamma_decay,
      seed = tune_seed_vec[j]
    )
  }

  # parallelise shard tuning RWs via mclapply
  shard_tune_list <- if (isTRUE(tune_parallel) && length(tune_ids) > 1L) {
    parallel::mclapply(seq_along(tune_ids), shard_tune_worker,
                       mc.cores = tune_mc_cores,
                       mc.allow.recursive = FALSE,
                       mc.preschedule = TRUE)
  } else {
    lapply(seq_along(tune_ids), shard_tune_worker)
  }

  # One step size for all shards. This assumes that the shards are
  # homogeneous, as in the simulated logistic regression, where every shard
  # has the same number of observations from the same model.
  # aggregate sigma (rw tuned step size) on the log scale into a vector
  shard_rwstep_log_scales <- vapply(shard_tune_list, function(x) log(x$sigma_final), numeric(1))
  # take the mean and bring it back to normal scale
  h_hat <- exp(mean(shard_rwstep_log_scales)) # this is your step size estimate
  # We us the geometric mean because step sizes are multiplicative objects,
  # so averaging on the log scale is the natural way.

  chain_seed_vec <- make_seed_vec(tune_seed_base + 30000L, nshards, name = "chain_seed", allow_null = FALSE)

  chain_worker <- function(i) {
    set.seed(chain_seed_vec[i])
    rwLogitChol(
      Y = dataY_list[[i]],
      X = dataX_list[[i]],
      xi = GLM_coef_list[[i]], # start from GLM coeffs
      sigma = h_hat, # use averaged adaptively tuned step size
      run = npart, # generate exactly npart particles
      chol_A = shard_metric_list[[i]]$chol # use shard-specific cholesky
    )
  }

  Chains_par <- if (isTRUE(chain_parallel) && nshards > 1L) {
    parallel::mclapply(seq_len(nshards), chain_worker,
                       mc.cores = chain_mc_cores,
                       mc.allow.recursive = FALSE,
                       mc.preschedule = TRUE)
  } else {
    lapply(seq_len(nshards), chain_worker)
  }

  # convert MCMC chains into NxCxd particle cubes
  Chains <- lapply(Chains_par, `[[`, "samples")
  Chains2 <- simplify2array(Chains)
  initPart <- aperm(Chains2, c(1, 3, 2))

  # pool shared data and calc GLM coefs and hessian
  dataX_tot <- do.call(rbind, dataX_list)
  dataY_tot <- unlist(dataY_list, use.names = FALSE)
  GLM <- stats::glm(dataY_tot ~ . - 1, family = stats::binomial(), data = as.data.frame(dataX_tot))
  GLM$coefficients <- safe_glm_coef(GLM, ndim) # "regularise" coefs

  full_metric <- laplace_rw_metric(GLM, ridge = laplace_ridge) # calculate the regularised Hessian for full dataset
  full_tune <- tune_logit_rw_scale( # tune RW step size on full data
    Y = dataY_tot,
    X = dataX_tot,
    init = GLM$coefficients,
    chol_metric = full_metric$chol,
    sigma_init = h_hat,
    target_acc = target_acc_rw,
    batch_size = tune_batch_size,
    n_batches = tune_n_batches,
    gamma0 = adapt_gamma0,
    gamma_decay = adapt_gamma_decay,
    seed = as.integer(tune_seed_base + 10000L)
  )
  h_hat_full <- full_tune$sigma_final

  if (compute_truth) {
    # divide Nmctruth between n_truth_chains -- it defaults to 1
    draws_per_truth_chain <- Nmctruth %/% n_truth_chains

    # create seed for truth chains
    truth_seed_vec <- make_seed_vec(tune_seed_base + 20000L, n_truth_chains, name = "tune_seed", allow_null = FALSE)

    # run the truth chain(s)
    truth_par <- parallel::mclapply(seq_len(n_truth_chains), function(i) {
      set.seed(truth_seed_vec[i])
      rwLogitChol(
        Y = dataY_tot, # full data
        X = dataX_tot,
        xi = GLM$coefficients,
        sigma = h_hat_full, # full parameters
        run = draws_per_truth_chain, # potentially a portion of Nmctruth
        chol_A = full_metric$chol # full parameters
      )
    }, mc.cores = n_truth_chains)

    truth_samples <- do.call(rbind, lapply(truth_par, `[[`, "samples")) # merge the parallel truth chains
    actualPart <- array(truth_samples, c(Nmctruth, 1L, ndim)) # this is a Nx1xd cube, so effectively a matrix
  } else {
    actualPart <- array(NA_real_, c(0L, 1L, ndim))
  }

  out <- list(
    Params = Params,
    initPart = initPart,
    actualPart = actualPart,
    dataX_list = dataX_list,
    dataY_list = dataY_list,
    GLM_coef_list = GLM_coef_list,
    dataX_tot = dataX_tot,
    dataY_tot = dataY_tot,
    GLM = GLM,
    GLM_list = GLM_list,
    shard_metric_list = shard_metric_list,
    full_metric = full_metric,
    h_hat = h_hat,
    h_hat_full = h_hat_full,
    shard_tune_list = shard_tune_list,
    full_tune = full_tune,
    tune_ids = tune_ids,
    tune_config = list(
      batch_size = tune_batch_size,
      n_batches = tune_n_batches,
      n_tune_shards = n_tune_shards,
      tune_seed = tune_seed,
      laplace_ridge = laplace_ridge,
      gamma0 = adapt_gamma0,
      gamma_decay = adapt_gamma_decay,
      target_acc_rw = target_acc_rw,
      UpperRW = UpperRW,
      compute_truth = compute_truth,
      tune_parallel = tune_parallel,
      tune_mc_cores = tune_mc_cores,
      chain_parallel = chain_parallel,
      chain_mc_cores = chain_mc_cores,
      safety_checks = safety_checks
    )
  )
  out
}


##################################################################
#### Distinct-row tables for the counted logit models.        ####
##################################################################
# phi depends on the data of a shard only through sums over its rows, so rows
# that repeat can be evaluated once and weighted. Both builders return one table
# shared by every shard, so that combining shards at a fusion node is addition
# of the count vectors.

logit_counts_row_keys <- function(X) {
  do.call(paste, c(as.data.frame(X), sep = "\r"))
}

# Distinct (y, x) rows with their multiplicities, as in Chan et al. (2023).
build_logit_counts_xypairs <- function(X_list, y_list) {
  if (!is.list(X_list) || !is.list(y_list) || length(X_list) != length(y_list)) {
    stop("X_list and y_list must be lists of the same length.", call. = FALSE)
  }

  C <- length(X_list)
  keys <- lapply(seq_len(C), function(c) {
    paste(y_list[[c]], logit_counts_row_keys(X_list[[c]]), sep = "\r")
  })
  X_all <- do.call(rbind, X_list)
  y_all <- unlist(y_list, use.names = FALSE)
  key_all <- unlist(keys, use.names = FALSE)

  first <- !duplicated(key_all)
  X_table <- X_all[first, , drop = FALSE]
  y_table <- y_all[first]
  levels_table <- key_all[first]
  # the keys are printed representations, so confirm they separate the rows
  if (nrow(X_table) != nrow(unique(cbind(y_all, X_all)))) {
    stop("Row keys did not separate the distinct (y, x) rows.", call. = FALSE)
  }

  count_list <- lapply(seq_len(C), function(c) {
    counts <- as.numeric(table(factor(keys[[c]], levels = levels_table)))
    if (sum(counts) != length(y_list[[c]])) {
      stop("Counted (y, x) rows do not account for every observation in a shard.", call. = FALSE)
    }
    counts
  })

  list(
    rep(list(X_table), C),
    rep(list(y_table), C),
    count_list
  )
}

######################################
#### Integrated Absolute Distance ####
######################################


integrated_absolute_distance <- function(full_post,
                                         fusion_post,
                                         bw = NULL,
                                         print_res = TRUE,
                                         grid_n = 10000L,
                                         full_weights = NULL,
                                         fusion_weights = NULL,
                                         return_details = FALSE) {
  as_sample_matrix <- function(x, name) {
    dims <- dim(x)
    if (is.null(dims)) {
      x <- matrix(as.numeric(x), ncol = 1L)
    } else if (length(dims) == 2L) {
      x <- as.matrix(x)
    } else if (length(dims) == 3L && dims[2L] == 1L) {
      x <- matrix(as.numeric(x[, 1L, , drop = FALSE]), nrow = dims[1L], ncol = dims[3L])
    } else {
      stop(sprintf("%s must be a vector, matrix, or N x 1 x d array.", name), call. = FALSE)
    }
    storage.mode(x) <- "double"
    if (!is.numeric(x) || any(!is.finite(x))) {
      stop(sprintf("%s must contain only finite numeric values.", name), call. = FALSE)
    }
    if (nrow(x) < 2L || ncol(x) < 1L) {
      stop(sprintf("%s must have at least two rows and one column.", name), call. = FALSE)
    }
    x
  }
  
  normalise_weights <- function(w, n, name) {
    if (is.null(w)) return(NULL)
    if (!is.numeric(w) || length(w) != n || any(!is.finite(w)) || any(w < 0)) {
      stop(sprintf("%s must be finite non-negative weights matching the number of rows.", name), call. = FALSE)
    }
    if (!any(w > 0)) {
      stop(sprintf("%s must sum to a positive finite value.", name), call. = FALSE)
    }
    log_w <- log(as.numeric(w))
    max_log_w <- max(log_w)
    log_sum_w <- max_log_w + log(sum(exp(log_w - max_log_w)))
    exp(log_w - log_sum_w)
  }
  
  recycle_bandwidth <- function(bw, dimensions) {
    if (is.null(bw)) {
      return(rep("nrd0", dimensions))
    }
    if (length(bw) == 1L) {
      return(rep(bw, dimensions))
    }
    if (length(bw) != dimensions) {
      stop("bw must have length 1 or ncol(full_post).", call. = FALSE)
    }
    bw
  }
  
  validate_grid_n <- function(grid_n) {
    if (!is.numeric(grid_n) || length(grid_n) != 1L || !is.finite(grid_n) || grid_n < 2) {
      stop("grid_n must be an integer >= 2.", call. = FALSE)
    }
    as.integer(grid_n)
  }
  
  trapz <- function(x, y) {
    sum(diff(x) * (head(y, -1L) + tail(y, -1L)) / 2)
  }
  
  kde <- function(x, weights, bw, from, to, grid_n) {
    args <- list(x = x, bw = bw, from = from, to = to, n = grid_n)
    if (!is.null(weights)) {
      args$weights <- weights
      args$warnWbw <- FALSE
    }
    do.call(stats::density, args)
  }
  
  print_bandwidths <- function(bandwidth) {
    cat("bandwidths chosen:", paste(signif(bandwidth, 4), collapse = ", "), "\n")
  }
  
  full_post <- as_sample_matrix(full_post, "full_post")
  is_many <- is.list(fusion_post) && is.null(dim(fusion_post))
  if (is_many && length(fusion_post) < 1L) {
    stop("fusion_post must be a sample matrix or a non-empty list of sample matrices.", call. = FALSE)
  }
  fusion_posts <- if (is_many) fusion_post else list(fusion_post)
  
  fusion_posts <- lapply(seq_along(fusion_posts), function(i) {
    name <- if (is_many) sprintf("fusion_post[[%s]]", i) else "fusion_post"
    x <- as_sample_matrix(fusion_posts[[i]], name)
    if (ncol(x) != ncol(full_post)) {
      stop("fusion_post must have ncol(full_post) columns.", call. = FALSE)
    }
    x
  })
  
  dimensions <- ncol(full_post)
  bw <- recycle_bandwidth(bw, dimensions)
  grid_n <- validate_grid_n(grid_n)
  
  full_weights <- normalise_weights(full_weights, nrow(full_post), "full_weights")
  if (is.null(fusion_weights)) {
    fusion_weights <- vector("list", length(fusion_posts))
  } else if (is_many) {
    if (!is.list(fusion_weights)) {
      if (length(fusion_posts) == 1L) {
        fusion_weights <- list(fusion_weights)
      } else {
        stop("fusion_weights must be NULL or a list matching fusion_post.", call. = FALSE)
      }
    }
    if (length(fusion_weights) != length(fusion_posts)) {
      stop("fusion_weights must be NULL or a list matching fusion_post.", call. = FALSE)
    }
  } else {
    fusion_weights <- if (is.list(fusion_weights)) fusion_weights else list(fusion_weights)
    if (length(fusion_weights) != 1L) {
      stop("fusion_weights must be NULL or a weight vector for a single fusion_post.", call. = FALSE)
    }
  }
  fusion_weights <- lapply(seq_along(fusion_posts), function(i) {
    normalise_weights(
      fusion_weights[[i]],
      nrow(fusion_posts[[i]]),
      if (is_many) sprintf("fusion_weights[[%s]]", i) else "fusion_weights"
    )
  })
  
  grid_from <- apply(full_post, 2L, min)
  grid_to <- apply(full_post, 2L, max)
  for (fusion_post in fusion_posts) {
    grid_from <- pmin(grid_from, apply(fusion_post, 2L, min))
    grid_to <- pmax(grid_to, apply(fusion_post, 2L, max))
  }
  
  baseline_kdes <- vector("list", dimensions)
  bandwidth_chosen <- numeric(dimensions)
  
  for (j in seq_len(dimensions)) {
    if (grid_from[j] == grid_to[j]) {
      bandwidth_chosen[j] <- NA_real_
      next
    }
    
    baseline_kdes[[j]] <- kde(
      x = full_post[, j],
      weights = full_weights,
      bw = bw[j],
      from = grid_from[j],
      to = grid_to[j],
      grid_n = grid_n
    )
    bandwidth_chosen[j] <- baseline_kdes[[j]]$bw
  }
  
  int_abs_dist <- matrix(0, nrow = length(fusion_posts), ncol = dimensions)
  
  for (i in seq_along(fusion_posts)) {
    for (j in seq_len(dimensions)) {
      if (grid_from[j] == grid_to[j]) {
        next
      }
      
      fusion_kde <- kde(
        x = fusion_posts[[i]][, j],
        weights = fusion_weights[[i]],
        bw = bandwidth_chosen[j],
        from = grid_from[j],
        to = grid_to[j],
        grid_n = grid_n
      )
      
      if (!isTRUE(all.equal(baseline_kdes[[j]]$x, fusion_kde$x))) {
        stop("The KDE grids for full_post and fusion_posts differ.", call. = FALSE)
      }
      
      int_abs_dist[i, j] <- trapz(
        baseline_kdes[[j]]$x,
        abs(baseline_kdes[[j]]$y - fusion_kde$y)
      )
    }
  }
  
  iad <- rowSums(int_abs_dist) / (2 * dimensions)
  
  if (isTRUE(print_res)) {
    print_bandwidths(bandwidth_chosen)
    if (is_many) {
      cat(sprintf("IAD range: %.6g to %.6g\n", min(iad), max(iad)))
    } else {
      cat(sprintf("IAD: %.6g\n", iad[1L]))
    }
  }
  
  if (isTRUE(return_details)) {
    if (!is_many) {
      return(list(
        IAD = iad[1L],
        per_dimension_total_variation = as.numeric(int_abs_dist[1L, ] / 2),
        integral_abs_distance = as.numeric(int_abs_dist[1L, ]),
        bandwidth = bandwidth_chosen,
        grid_from = grid_from,
        grid_to = grid_to
      ))
    }
    
    return(list(
      IAD = iad,
      per_run_per_dimension_total_variation = int_abs_dist / 2,
      integral_abs_distance = int_abs_dist,
      bandwidth = bandwidth_chosen,
      grid_from = grid_from,
      grid_to = grid_to
    ))
  }
  
  if (is_many) iad else iad[1L]
}


consensus_mc <- function(samples) {
  if (is.array(samples) && length(dim(samples)) == 3L) {
    samples <- lapply(seq_len(dim(samples)[2]), function(shard) samples[, shard, ])
  }
  
  n <- nrow(samples[[1]])
  d <- ncol(samples[[1]])
  
  for (shard in seq_along(samples)) {
    if (!all(dim(samples[[shard]]) == c(n, d))) {
      stop("All shard sample matrices must have the same dimensions.", call. = FALSE)
    }
  }
  
  precision_list <- lapply(samples, function(x) solve(stats::cov(x)))
  precision_sum <- Reduce(`+`, precision_list)
  precision_sum_inv <- solve(precision_sum)
  
  combined <- matrix(0, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    numerator <- matrix(0, nrow = d, ncol = 1)
    for (shard in seq_along(samples)) {
      numerator <- numerator + precision_list[[shard]] %*% samples[[shard]][i, ]
    }
    combined[i, ] <- as.vector(precision_sum_inv %*% numerator)
  }
  
  colnames(combined) <- colnames(samples[[1]])
  combined
}


####
