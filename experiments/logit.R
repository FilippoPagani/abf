################################################################################
#### Simulated logistic regression: ABF, BF and CMC                         ####
################################################################################
# Section "Comparison on Bayesian logistic regression" of the paper: the
# results of Figure fig:logistic_iad_compare and Table
# tab:logistic_iad_comparison and, with the settings given below, of Figure
# fig:step_size_compare (Appendix "Step size comparison").
#
# The data are 4 shards of 500 observations from the logistic regression of the
# paper, with a flat prior. Random-Walk Metropolis gives the shard draws and the
# reference sample. For each target zeta in abf_zeta_vec (ABF) and bf_zeta_vec
# (BF), the coalescence time T is chosen by rejection sampling and the number
# of steps n by the bisection on the one-step ESS, then nRep replicas are run.
# CMC aggregates the same shard draws, permuted at random, nRep times.
#
# Output: results_file, with one row per run (method, zeta, n, runtime, IAD),
# the tuning of each zeta, and the steps of every ABF run. Draw the figures
# with plotting/plot_logit_iad.R and plotting/plot_logit_step_size_rules.R.
# The script also prints the table.
#
# Run time: a few hours with the settings below, mostly for BF; ABF and CMC
# alone take minutes. It uses up to 10 cores.
#
# Variants:
#  - Figure fig:step_size_compare: set abf_zeta_vec <- 0.4,
#    bf_zeta_vec <- numeric(0), step_size_min_factor <- 1e-8 and
#    step_size_max_factor <- 1e8 (so that the bounds never bind), and run the
#    script twice: with step_size_criterion <- "ess_ratio" and
#    results_file <- "results/logit/step_size_cess.rds", and with
#    step_size_criterion <- "asymptotic" and
#    results_file <- "results/logit/step_size_asymptotic.rds".
#  - Adaptive mesh: step_size_min_factor <- 0.5, step_size_max_factor <- 1.5.
#    Then T / n is only the first step.
#  - Preconditioning of ABF: preconditioning <- "diagonal" or "full".

library(this.path)
setwd(this.path::here(.. = 1))
if (!requireNamespace("layeredBB", quietly = TRUE)) {
  stop("Package 'layeredBB' is required for exact Bayesian Fusion.", call. = FALSE)
}
source("R/common_setup.R") # this clears the workspace

total_start <- Sys.time()

## Settings ##
model_name <- "logit"
N <- 10000L # particles of ABF and BF, and shard draws
NMCtruth <- 200000L # reference sample
nShards <- 4L
nData <- 500L # observations per shard
ndim <- 6L
nRep <- 10L

abf_zeta_vec <- c(0.1, 0.2, 0.5, 0.8, 0.9, 0.99)
bf_zeta_vec <- c(0.1, 0.2, 0.3, 0.4) # numeric(0) skips BF
# Both factors equal to 1 keep the step T / n for the whole ABF run.
step_size_min_factor <- 1
step_size_max_factor <- 1
step_size_criterion <- "ess_ratio" # or "asymptotic"
preconditioning <- "no" # or "diagonal" or "full"
resamp_thr <- 0.5

rw_tune_batch_size <- 100L
rw_tune_n_batches <- 100L
n_tuning_particles <- min(N, 2000L)
n_eval_reps <- 4L # one-step SMC runs per evaluation in the bisection on n
replicate_mc_cores <- min(nRep, 10L)
iad_grid_n <- 10000L
results_file <- "results/logit/logit.rds"

eff_seed <- make_session_seed(17855)

## Data, shard draws and reference sample ##
beta <- rep(c(-1, 1), length.out = ndim) / sqrt(ndim)
logit_setup <- generate_logit_data_and_shards(
  true_betas = beta,
  ndata = nData,
  nshards = nShards,
  npart = N,
  Nmctruth = NMCtruth,
  UpperRW = 1 / length(beta),
  tune_batch_size = rw_tune_batch_size,
  tune_n_batches = rw_tune_n_batches,
  n_tune_shards = nShards,
  tune_seed = eff_seed + 500L,
  compute_truth = TRUE,
  tune_parallel = TRUE,
  tune_mc_cores = nShards
)
Params <- logit_setup$Params
initPart <- logit_setup$initPart
actualPart <- logit_setup$actualPart

## ABF and BF ##
# The same eff_seed in every call gives the same T and the same replica seeds,
# so the replicas start from the same rejection-sampled particles for every zeta.
run_fusion <- function(zeta, ...) {
  cat(sprintf("Running zeta = %s.\n", zeta))
  run_tuned_smc_experiment(
    n_rep = nRep,
    n_particles = N,
    nshards = nShards,
    ndim = ndim,
    model_name = model_name,
    params = Params,
    init_particles = initPart,
    eff_seed = eff_seed,
    resamp_thr = resamp_thr,
    target_one_step_ess = zeta,
    n_eval_reps = n_eval_reps,
    n_tuning_particles = n_tuning_particles,
    replicate_mc_cores = replicate_mc_cores,
    ...
  )
}

abf_results <- lapply(
  abf_zeta_vec,
  run_fusion,
  eval_mc_cores = 1L,
  preconditioning = preconditioning,
  step_size_min_factor = step_size_min_factor,
  step_size_max_factor = step_size_max_factor,
  step_size_criterion = step_size_criterion
)
bf_results <- lapply(
  bf_zeta_vec,
  run_fusion,
  eval_mc_cores = min(n_eval_reps, replicate_mc_cores),
  nchunks_initial_guess = if (length(abf_results) > 0L) abf_results[[1]]$nChunks_hat,
  smc_function = bayesian_fusion_smc_lbb_poisson
)

## CMC ##
make_cmc_replica <- function(rep_idx) {
  set.seed(eff_seed + 7000L + rep_idx)
  cmc_start <- Sys.time()
  cmc_input <- initPart
  for (shard_idx in seq_len(nShards)) {
    cmc_input[, shard_idx, ] <- initPart[sample.int(N), shard_idx, ]
  }
  cmc_out <- consensus_mc(cmc_input)
  list(out = cmc_out, elapsed_seconds = as.numeric(Sys.time() - cmc_start, units = "secs"))
}
cmc_runs <- lapply(seq_len(nRep), make_cmc_replica)

## One row per run, and the IAD of every run ##
posterior_samples <- list()
posterior_weights <- list()
plot_rows <- list()
add_run <- function(method, zeta, run_result, run, n_steps) {
  row_idx <- length(posterior_samples) + 1L
  posterior_samples[[row_idx]] <<- run$out
  posterior_weights[row_idx] <<- list(run$normw)
  plot_rows[[row_idx]] <<- data.frame(
    method = method,
    zeta = zeta,
    T = if (is.null(run_result)) NA_real_ else run_result$endT_hat,
    nSteps = n_steps,
    Delta = if (is.null(run_result)) NA_real_ else run_result$endT_hat / run_result$nChunks_hat,
    rep_idx = run$rep_idx,
    elapsed_seconds = run$elapsed_seconds
  )
}
for (i in seq_along(abf_results)) {
  for (run in abf_results[[i]]$runs) add_run("ABF", abf_zeta_vec[i], abf_results[[i]], run, run$n_total_steps)
}
for (i in seq_along(bf_results)) {
  for (run in bf_results[[i]]$runs) add_run("BF", bf_zeta_vec[i], bf_results[[i]], run, run$nchunks)
}
for (rep_idx in seq_len(nRep)) {
  add_run("CMC", NA_real_, NULL, c(cmc_runs[[rep_idx]], rep_idx = rep_idx), NA_real_)
}

plot_data <- do.call(rbind, plot_rows)
plot_data$IAD <- integrated_absolute_distance(
  full_post = actualPart,
  fusion_post = posterior_samples,
  print_res = FALSE,
  grid_n = iad_grid_n,
  fusion_weights = posterior_weights
)

# The tuning of each zeta: T, n, and the estimated one-step ESS at n.
all_results <- c(abf_results, bf_results)
setup_summary <- data.frame(
  method = c(rep("ABF", length(abf_results)), rep("BF", length(bf_results))),
  zeta = c(abf_zeta_vec, bf_zeta_vec),
  T = vapply(all_results, function(x) x$endT_hat, numeric(1)),
  nSteps = vapply(all_results, function(x) x$nChunks_hat, numeric(1)),
  zeta_hat = vapply(all_results, function(x) x$zeta_hat, numeric(1)),
  setup_seconds = vapply(all_results, function(x) x$tuning_elapsed_seconds, numeric(1))
)

# The steps of every ABF run: the step taken, its CESS ratio, and the steps
# that the CESS rule and the asymptotic rule recommend for the next transition.
steps <- do.call(rbind, lapply(seq_along(abf_results), function(i) {
  do.call(rbind, lapply(abf_results[[i]]$runs, function(run) {
    data.frame(
      zeta = abf_zeta_vec[i],
      rep_idx = run$rep_idx,
      time = run$time_after_step,
      delta_used = run$delta_used,
      conditional_ess_ratio = run$conditional_ess_ratio,
      delta_ess = run$delta_ess_raw,
      delta_asymptotic = run$delta_asymptotic_raw
    )
  }))
}))

dir.create(dirname(results_file), showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  plot_data = plot_data,
  setup_summary = setup_summary,
  steps = steps,
  step_size_criterion = step_size_criterion,
  end_time = if (length(abf_results) > 0L) abf_results[[1]]$endT_hat else NA_real_
), results_file)

## Table tab:logistic_iad_comparison ##
method_summary <- do.call(rbind, lapply(
  split(plot_data, list(plot_data$method, plot_data$zeta), drop = TRUE),
  function(x) data.frame(
    method = x$method[1L], zeta = x$zeta[1L], n_steps = x$nSteps[1L],
    IAD_mean = mean(x$IAD), IAD_min = min(x$IAD), IAD_max = max(x$IAD),
    time_mean = mean(x$elapsed_seconds), time_min = min(x$elapsed_seconds),
    time_max = max(x$elapsed_seconds)
  )
))
cmc_rows <- plot_data[plot_data$method == "CMC", ]
method_summary <- rbind(method_summary, data.frame(
  method = "CMC", zeta = NA, n_steps = NA,
  IAD_mean = mean(cmc_rows$IAD), IAD_min = min(cmc_rows$IAD), IAD_max = max(cmc_rows$IAD),
  time_mean = mean(cmc_rows$elapsed_seconds), time_min = min(cmc_rows$elapsed_seconds),
  time_max = max(cmc_rows$elapsed_seconds)
))
row.names(method_summary) <- NULL

cat("\nTuning for each method and zeta (T is the same for all, as the seed is):\n")
print(setup_summary, row.names = FALSE, digits = 6)
cat("\nIAD and runtime per replica (mean, min, max):\n")
print(method_summary, row.names = FALSE, digits = 4)
cat(sprintf("Results saved to %s.\n", results_file))
print(Sys.time() - total_start)
