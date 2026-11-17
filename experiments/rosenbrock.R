################################################################################
#### 5d hybrid Rosenbrock target: i.i.d. sample, D&C ABF and CMC            ####
################################################################################
# Section "Rosenbrock distribution, 5d Banana" of the paper: the results of
# Figure fig:rosen5d_cmc_compare.
#
# The target is split into C = 4 subposteriors of the same form, each with
# 1/C of the exponent of the target. Exact i.i.d. samplers give the
# subposterior draws and the reference sample. ABF fuses the subposteriors on
# a binary tree; CMC aggregates the same draws.
#
# Output: results_file, with the reference sample, the ABF particles and
# weights, and the CMC sample. Draw the figure with plotting/plot_rosenbrock.R.
# The script also prints the mean and standard deviation of each coordinate for
# the three samples.
#
# Run time: about an hour with N = 10^6 on 8 cores; with N = 2 * 10^4, a few
# minutes.
#
# Variants:
#  - Step size: change target_one_step_ess; tune_nchunks <- TRUE takes the
#    first step from the bisection on the one-step ESS instead of the spread of
#    phi; step_size_criterion <- "asymptotic" uses the asymptotic rule.
#  - Preconditioning: preconditioning <- "no" or "diagonal".
#  - Fixed grid: use_offline_grid <- TRUE runs each fusion first as an adaptive
#    pilot, then reruns it on the pilot's grid with each step split into
#    offline_grid_refine parts.

library(this.path)
setwd(this.path::here(.. = 1))
source("R/common_setup.R") # this clears the workspace

total_start <- Sys.time()

## Settings ##
model_name <- "rosen5d"
N <- 1000000L # particles, and draws from each subposterior
NMCtruth <- 1000000L # reference sample
nShards <- 4L # a power of two
ndim <- 5L
nCores <- 8L # threads per fusion

target_one_step_ess <- 0.99
tune_nchunks <- FALSE
step_size_criterion <- "ess_ratio"
step_size_min_factor <- 0.5
step_size_max_factor <- 1.5
step_size_total_min <- 1e-6
preconditioning <- "full"
use_offline_grid <- FALSE
offline_grid_refine <- 1L
target_rejection_acceptance <- 0.01
resamp_thr <- 0.5
n_tuning_particles <- 100000L # used only if tune_nchunks is TRUE
n_eval_reps <- 10L # used only if tune_nchunks is TRUE
results_file <- "results/rosenbrock/rosenbrock.rds"

eff_seed <- make_session_seed(18193)

# Parameters (a, b12, b13, b22, b23, mu) of the target and of one subposterior,
# which has 1/nShards of each exponent.
a <- 40 / 20 / nShards
b <- 80 / 20 / nShards
mu <- 0
Params <- list(a, b, b, b, b, mu)
actualParams <- c(lapply(Params[-length(Params)], function(x) x * nShards), list(mu))

initPart <- generate_init_particles_rosen5d(npart = N, nshards = nShards, ndim = ndim, params = Params)
actualPart <- generate_init_particles_rosen5d(npart = NMCtruth, nshards = 1L, ndim = ndim, params = actualParams)

## D&C ABF ##
leaf_samples <- lapply(seq_len(nShards), function(shard) {
  matrix(initPart[, shard, , drop = FALSE], nrow = N, ncol = ndim)
})
dc_run <- run_dc_abf(
  leaf_samples = leaf_samples,
  model_name = model_name,
  params = Params,
  n_particles = N,
  n_tuning_particles = n_tuning_particles,
  seed = eff_seed + 1000L,
  n_cores = nCores,
  target_rejection_acceptance = target_rejection_acceptance,
  target_one_step_ess = target_one_step_ess,
  n_eval_reps = n_eval_reps,
  resamp_thr = resamp_thr,
  preconditioning = preconditioning,
  step_size_min_factor = step_size_min_factor,
  step_size_max_factor = step_size_max_factor,
  step_size_total_min = step_size_total_min,
  step_size_criterion = step_size_criterion,
  use_offline_grid = use_offline_grid,
  offline_grid_refine = offline_grid_refine,
  tune_nchunks = tune_nchunks
)

## CMC ##
cmc_start <- Sys.time()
cmc_result <- consensus_mc(initPart)
cmc_elapsed_seconds <- as.numeric(Sys.time() - cmc_start, units = "secs")

fusion_out <- dc_run$root$out
fusion_weights <- dc_run$root$normw[, 1L] / sum(dc_run$root$normw)

dir.create(dirname(results_file), showWarnings = FALSE, recursive = TRUE)
saveRDS(list(
  truth = matrix(actualPart, nrow = NMCtruth),
  abf = fusion_out,
  abf_weights = fusion_weights,
  cmc = cmc_result,
  eff_seed = eff_seed
), results_file)

## Summaries ##
node_summary <- do.call(rbind, lapply(seq.int(2L, length(dc_run$levels)), function(level_idx) {
  do.call(rbind, lapply(dc_run$levels[[level_idx]], function(node) {
    data.frame(
      level = level_idx,
      shards = paste(node$leaf_ids, collapse = ","),
      T = node$diagnostics$endT_fit$end_time,
      n_steps = node$diagnostics$smc$n_total_steps,
      min_step = min(node$diagnostics$smc$delta_used),
      max_step = max(node$diagnostics$smc$delta_used),
      resampcount = node$diagnostics$smc$resampcount,
      elapsed_seconds = node$diagnostics$elapsed_seconds
    )
  }))
}))
weighted_moments <- function(x, w = rep(1 / nrow(x), nrow(x))) {
  m <- colSums(x * w)
  rbind(mean = m, sd = sqrt(colSums(sweep(x, 2L, m)^2 * w)))
}
moments <- rbind(
  iid = weighted_moments(matrix(actualPart, nrow = NMCtruth)),
  ABF = weighted_moments(fusion_out, fusion_weights),
  CMC = weighted_moments(cmc_result)
)
rownames(moments) <- paste(rep(c("iid", "ABF", "CMC"), each = 2L), rownames(moments))
colnames(moments) <- paste0("theta", seq_len(ndim))

cat("\nFusions of the tree:\n")
print(node_summary, row.names = FALSE, digits = 4)
cat(sprintf("\nABF: %.1f seconds, final ESS %.0f. CMC: %.2f seconds.\n",
            dc_run$elapsed_seconds, 1 / sum(fusion_weights^2), cmc_elapsed_seconds))
cat("\nMean and standard deviation of each coordinate:\n")
print(round(moments, 3))
cat(sprintf("Results saved to %s.\n", results_file))
print(Sys.time() - total_start)
