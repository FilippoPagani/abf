# A single fusion of all the shards, without the divide-and-conquer tree, run
# for several replicas. Used by experiments/logit.R for ABF and exact BF.

run_tuned_smc_experiment <- function(n_rep,
                                     n_particles,
                                     nshards,
                                     ndim,
                                     model_name,
                                     params,
                                     init_particles,
                                     eff_seed,
                                     align_phi_on_resample = TRUE,
                                     resamp_flag = TRUE,
                                     resamp_thr = 0.5,
                                     target_rejection_acceptance = 0.01,
                                     target_one_step_ess = 0.9,
                                     base_step_size = 0.01,
                                     n_eval_reps = 10L,
                                     n_tuning_particles = n_particles,
                                     eval_mc_cores = 1L,
                                     replicate_mc_cores = n_rep,
                                     safety_checks = TRUE,
                                     verbose = TRUE,
                                     smc_function = bayesian_fusion_smc,
                                     nchunks_initial_guess = NULL,
                                     preconditioning = c("no", "diagonal", "full"),
                                     preconditioning_ridge = 1e-6,
                                     preconditioning_max_tries = 8L,
                                     step_size_min_factor = 0.5,
                                     step_size_max_factor = 1.5,
                                     step_size_total_min = 0,
                                     step_size_criterion = "ess_ratio") {
  # Runs one fusion of all the shards (no divide-and-conquer tree) n_rep times.
  # It chooses the coalescence time T by rejection sampling, then the number of
  # steps n by the bisection on the one-step ESS, and runs n_rep replicas in
  # parallel from independent rejection-sampled particles. smc_function is
  # bayesian_fusion_smc (ABF) or bayesian_fusion_smc_lbb_poisson (exact BF).
  # With step_size_min_factor = step_size_max_factor = 1, ABF keeps the step
  # T / n for the whole run; otherwise T / n is only the first step.

  controller_argument_supplied <- c(
    step_size_criterion = !missing(step_size_criterion),
    step_size_min_factor = !missing(step_size_min_factor),
    step_size_max_factor = !missing(step_size_max_factor),
    step_size_total_min = !missing(step_size_total_min)
  )
  is_approximate_abf <- identical(smc_function, bayesian_fusion_smc)
  is_lbb <- identical(smc_function, bayesian_fusion_smc_lbb_poisson)
  if (is_lbb && any(controller_argument_supplied)) {
    stop("online adaptive step-size controls are unsupported for bayesian_fusion_smc_lbb_poisson().", call. = FALSE)
  }

  integer_max <- .Machine$integer.max
  if (!is.numeric(eff_seed) || length(eff_seed) != 1L || !is.finite(eff_seed) ||
      eff_seed < 1 || eff_seed != floor(eff_seed) || eff_seed > integer_max) {
    stop("eff_seed must be a positive integer within R's integer range.", call. = FALSE)
  }
  if (!is.numeric(n_rep) || length(n_rep) != 1L || !is.finite(n_rep) ||
      n_rep < 1 || n_rep != floor(n_rep) || n_rep > integer_max) {
    stop("n_rep must be a positive integer within R's integer range.", call. = FALSE)
  }
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L ||
      !is.finite(n_eval_reps) || n_eval_reps < 1 ||
      n_eval_reps != floor(n_eval_reps) || n_eval_reps > integer_max) {
    stop("n_eval_reps must be a positive integer within R's integer range.", call. = FALSE)
  }

  # Seeds: eff_seed + 1000 for T, eff_seed + 2000 + (0, 1, ...) for the
  # bisection on n, eff_seed + 4000 + (0, 1, ...) for the ESS at the chosen n,
  # and eff_seed + 5000 + (0, ..., n_rep - 1) for the replicas.
  seed_bounds <- c(
    eff_seed + 1000,
    eff_seed + 2000 + n_eval_reps - 1,
    eff_seed + 4000 + n_eval_reps - 1,
    eff_seed + 5000 + n_rep - 1
  )
  if (any(!is.finite(seed_bounds)) || any(seed_bounds < 1) ||
      any(seed_bounds != floor(seed_bounds)) || any(seed_bounds > integer_max)) {
    stop("eff_seed-derived seed values must be positive integers within R's integer range.", call. = FALSE)
  }

  if (isTRUE(safety_checks)) {
    n_rep <- as_positive_int(n_rep, "n_rep")
    n_particles <- as_positive_int(n_particles, "n_particles")
    nshards <- as_positive_int(nshards, "nshards")
    ndim <- as_positive_int(ndim, "ndim")
    validate_particle_cube(init_particles, nshards = nshards, ndim = ndim)
  } else {
    n_rep <- as.integer(n_rep)
    n_particles <- as.integer(n_particles)
    nshards <- as.integer(nshards)
    ndim <- as.integer(ndim)
  }

  if (!is.numeric(replicate_mc_cores) || length(replicate_mc_cores) != 1L || !is.finite(replicate_mc_cores) || replicate_mc_cores < 1) stop("replicate_mc_cores must be a positive integer.", call. = FALSE)
  if (!is.numeric(target_rejection_acceptance) || length(target_rejection_acceptance) != 1L || !is.finite(target_rejection_acceptance) || target_rejection_acceptance <= 0 || target_rejection_acceptance >= 1) stop("target_rejection_acceptance must lie in (0, 1).", call. = FALSE)
  if (!is.numeric(target_one_step_ess) || length(target_one_step_ess) != 1L || !is.finite(target_one_step_ess) || target_one_step_ess <= 0 || target_one_step_ess >= 1) stop("target_one_step_ess must lie in (0, 1).", call. = FALSE)
  if (!is.numeric(step_size_min_factor) || length(step_size_min_factor) != 1L || !is.finite(step_size_min_factor) ||
      !is.numeric(step_size_max_factor) || length(step_size_max_factor) != 1L || !is.finite(step_size_max_factor) ||
      step_size_min_factor <= 0 || step_size_min_factor > 1 || step_size_max_factor < 1) {
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
  if (!is.numeric(base_step_size) || length(base_step_size) != 1L || !is.finite(base_step_size) || base_step_size <= 0) stop("base_step_size must be positive.", call. = FALSE)
  if (!is.null(nchunks_initial_guess) &&
      (!is.numeric(nchunks_initial_guess) || length(nchunks_initial_guess) != 1L ||
       !is.finite(nchunks_initial_guess) || nchunks_initial_guess < 1 ||
       nchunks_initial_guess != floor(nchunks_initial_guess))) {
    stop("nchunks_initial_guess must be a positive integer or NULL.", call. = FALSE)
  }
  if (!is.numeric(n_eval_reps) || length(n_eval_reps) != 1L || !is.finite(n_eval_reps) || n_eval_reps < 1 || n_eval_reps != floor(n_eval_reps)) stop("n_eval_reps must be a positive integer.", call. = FALSE)
  if (!is.numeric(n_tuning_particles) || length(n_tuning_particles) != 1L || !is.finite(n_tuning_particles) || n_tuning_particles < 1 || n_tuning_particles != floor(n_tuning_particles)) stop("n_tuning_particles must be a positive integer.", call. = FALSE)
  if (!is.numeric(eval_mc_cores) || length(eval_mc_cores) != 1L || !is.finite(eval_mc_cores) || eval_mc_cores < 1 || eval_mc_cores != floor(eval_mc_cores)) stop("eval_mc_cores must be a positive integer.", call. = FALSE)
  if (!is.function(smc_function)) stop("smc_function must be a function.", call. = FALSE)
  preconditioning <- match.arg(preconditioning)
  if (!is.numeric(preconditioning_ridge) || length(preconditioning_ridge) != 1L ||
      !is.finite(preconditioning_ridge) || preconditioning_ridge < 0) {
    stop("preconditioning_ridge must be a finite non-negative number.", call. = FALSE)
  }
  if (!is.numeric(preconditioning_max_tries) || length(preconditioning_max_tries) != 1L ||
      !is.finite(preconditioning_max_tries) || preconditioning_max_tries < 1 ||
      preconditioning_max_tries != floor(preconditioning_max_tries)) {
    stop("preconditioning_max_tries must be a positive integer.", call. = FALSE)
  }
  if (preconditioning != "no" && !is_approximate_abf) {
    stop("Empirical preconditioning is supported only by bayesian_fusion_smc().", call. = FALSE)
  }
  subposterior_means <- NULL
  subposterior_variance <- NULL
  if (is_approximate_abf) {
    init_npart <- dim(init_particles)[1L]
    subposterior_means <- matrix(NA_real_, nrow = nshards, ncol = ndim)
    # s^2 = C b / m of the asymptotic step-size rule: the average variance of
    # the coordinates of the subposteriors.
    subposterior_variance <- 0
    for (shard in seq_len(nshards)) {
      theta_rows <- matrix(init_particles[, shard, ], nrow = init_npart, ncol = ndim)
      subposterior_means[shard, ] <- colMeans(theta_rows)
      subposterior_variance <- subposterior_variance + sum(apply(theta_rows, 2L, var))
    }
    subposterior_variance <- subposterior_variance / (nshards * ndim)
    if (!identical(dim(subposterior_means), c(nshards, ndim))) {
      stop("subposterior_means must have dimensions nshards by ndim.", call. = FALSE)
    }
    if (any(!is.finite(subposterior_means))) {
      stop("subposterior_means must contain only finite values.", call. = FALSE)
    }
  }

  eff_seed <- as.integer(eff_seed)
  replicate_mc_cores <- as.integer(replicate_mc_cores)
  n_eval_reps <- as.integer(n_eval_reps)
  n_tuning_particles <- as.integer(n_tuning_particles)
  eval_mc_cores <- min(as.integer(eval_mc_cores), n_eval_reps)
  if (!is.null(nchunks_initial_guess)) nchunks_initial_guess <- as.integer(nchunks_initial_guess)
  preconditioning_max_tries <- as.integer(preconditioning_max_tries)
  ## ##

  tuning_start <- Sys.time()

  if (isTRUE(verbose)) {
    cat(sprintf("eff_seed: %s\n", eff_seed))
  }

  max_iter_rejs <- max(n_particles, n_tuning_particles) / target_rejection_acceptance * 10
  n_rejection_proposals <- as.integer(n_particles * 10)
  abf_preconditioner <- if (is_approximate_abf && preconditioning != "no") {
    prepare_abf_preconditioner(
      init_particles = init_particles,
      preconditioning = preconditioning,
      ridge = preconditioning_ridge,
      max_tries = preconditioning_max_tries
    )
  } else {
    NULL
  }

  set.seed(eff_seed + 1000)

  # find endT_hat so that the rejection sampler has the desired average acceptance probability.
  if (is_approximate_abf) {
    endT_fit <- tune_rejection_end_time_cpp(
      init_particles = init_particles,
      n_samples = n_rejection_proposals,
      target_acceptance = target_rejection_acceptance,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner
    )
  } else {
    endT_fit <- tune_rejection_end_time_cpp(
      init_particles = init_particles,
      n_samples = n_rejection_proposals,
      target_acceptance = target_rejection_acceptance
    )
  }
  endT_hat <- endT_fit$end_time
  if (isTRUE(verbose)) {
    cat(sprintf("endT_hat = %.4f\n", endT_hat))
  }

  if (is.null(nchunks_initial_guess)) {
    nchunks_initial_guess <- max(
      1L,
      ceiling(endT_hat / base_step_size * sqrt(ndim))
    )
  }

  # generate a cube of resampled initial particles
  # only used for finding nChunks_hat (hence "tuning")
  if (is_approximate_abf) {
    tuning_particles <- rej_samp_init_part(
      out_length_tot = n_tuning_particles,
      end_time = endT_hat,
      init_part = init_particles,
      max_iter = max_iter_rejs,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner
    )
  } else {
    tuning_particles <- rej_samp_init_part(
      out_length_tot = n_tuning_particles,
      end_time = endT_hat,
      init_part = init_particles,
      max_iter = max_iter_rejs
    )
  }

  # find the nChunks_hat that drops ESS by desired amount on 1st smc step
  if (is_approximate_abf) {
    nchunks_fit <- tune_nchunks_from_one_step_ess(
      target_ess = target_one_step_ess, # target ess here, e.g. 0.7
      model = model_name,
      end_time = endT_hat,
      init_particles = tuning_particles,
      params = params,
      initial_guess = nchunks_initial_guess,
      n_eval_reps = n_eval_reps,
      eval_mc_cores = eval_mc_cores,
      seed = eff_seed + 2000,
      base_step_size = base_step_size,
      resamp = resamp_flag,
      resamp_thr = resamp_thr,
      align_phi_on_resample = align_phi_on_resample,
      smc_function = smc_function,
      preconditioning = preconditioning,
      abf_preconditioner = abf_preconditioner
    )
  } else {
    nchunks_fit <- tune_nchunks_from_one_step_ess(
      target_ess = target_one_step_ess, # target ess here, e.g. 0.7
      model = model_name,
      end_time = endT_hat,
      init_particles = tuning_particles,
      params = params,
      initial_guess = nchunks_initial_guess,
      n_eval_reps = n_eval_reps,
      eval_mc_cores = eval_mc_cores,
      seed = eff_seed + 2000,
      base_step_size = base_step_size,
      resamp = resamp_flag,
      resamp_thr = resamp_thr,
      align_phi_on_resample = align_phi_on_resample,
      smc_function = smc_function
    )
  }
  nChunks_hat <- nchunks_fit$nchunks
  zeta_hat <- nchunks_fit$ess_estimate

  tuning_elapsed_seconds <- as.numeric(Sys.time() - tuning_start, units = "secs")

  if (isTRUE(verbose)) {
    cat(sprintf("nChunks_hat = %s\n", nChunks_hat))
    cat(sprintf("Delta_hat = %.6g, estimated zeta = %.6g\n", endT_hat / nChunks_hat, zeta_hat))
  }
  rm(tuning_particles)

  # parallel replicas
  run_start <- Sys.time()
  worker_seed_vec <- make_seed_vec(eff_seed + 5000, n_rep, name = "replicate_seed", allow_null = FALSE)
  runs <- parallel::mclapply(
    seq_len(n_rep),
    function(wid) {
      set.seed(worker_seed_vec[wid])
      worker_start <- Sys.time()

      # rejection-sampled initial particles of this replica
      if (is_approximate_abf) {
        init_particles_rep <- rej_samp_init_part(
          out_length_tot = n_particles,
          end_time = endT_hat,
          init_part = init_particles,
          max_iter = max_iter_rejs,
          preconditioning = preconditioning,
          abf_preconditioner = abf_preconditioner
        )
        smc_out <- smc_function(
          end_time = endT_hat,
          nchunks = nChunks_hat,
          model = model_name,
          init_particles = init_particles_rep,
          params = params,
          init_weight = "n",
          resamp = resamp_flag,
          resamp_thr = resamp_thr,
          align_phi_on_resample = align_phi_on_resample,
          preconditioning = preconditioning,
          abf_preconditioner = abf_preconditioner,
          target_one_step_ess = target_one_step_ess,
          step_size_min_factor = step_size_min_factor,
          step_size_max_factor = step_size_max_factor,
          step_size_total_min = step_size_total_min,
          step_size_criterion = step_size_criterion,
          subposterior_means = subposterior_means,
          subposterior_variance = subposterior_variance
        )
      } else {
        init_particles_rep <- rej_samp_init_part(
          out_length_tot = n_particles,
          end_time = endT_hat,
          init_part = init_particles,
          max_iter = max_iter_rejs
        )
        smc_out <- smc_function(
          end_time = endT_hat,
          nchunks = nChunks_hat,
          model = model_name,
          init_particles = init_particles_rep,
          params = params,
          init_weight = "n",
          resamp = resamp_flag,
          resamp_thr = resamp_thr,
          align_phi_on_resample = align_phi_on_resample
        )
        smc_out$nchunks <- nChunks_hat
      }
      smc_out$elapsed_seconds <- as.numeric(Sys.time() - worker_start, units = "secs")
      smc_out$rep_idx <- wid
      smc_out
    },
    mc.cores = replicate_mc_cores,
    mc.allow.recursive = FALSE,
    mc.preschedule = TRUE
  )
  batch_elapsed_seconds <- as.numeric(Sys.time() - run_start, units = "secs")

  list(
    endT_hat = endT_hat,
    nChunks_hat = nChunks_hat,
    zeta_hat = zeta_hat,
    tuning_elapsed_seconds = tuning_elapsed_seconds,
    batch_elapsed_seconds = batch_elapsed_seconds,
    runs = runs,
    endT_fit = endT_fit,
    nchunks_fit = nchunks_fit,
    abf_preconditioner = abf_preconditioner
  )
}
