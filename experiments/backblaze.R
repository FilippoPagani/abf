################################################################################
#### Weibull proportional hazards model on the Backblaze data: D&C ABF, CMC ####
################################################################################
# Section "Proportional hazards model" and Appendix "Details of the
# proportional hazards model" of the paper: the results of Figures fig:ph_iad
# and fig:ph_pairs and Table tab:ph_zeta_N and, with the settings given below,
# of Table tab:ph_finegrid.
#
# The data are the hds data set of the R package dynamichazard (Backblaze hard
# drives), downloaded at a fixed commit. One record is kept per drive, for the
# G = 4 drive families with the most drives, and the drives of each family and
# failure status are spread evenly over C = 16 shards. The model has a random
# effect for each family, in centred coordinates
# q = (beta_0, beta_1, log gamma, log omega, u_1, ..., u_4); in the code these
# are (alpha, beta, log_rho, log_omega, u). The script
#  1. builds the data and the shards;
#  2. samples each shard with Stan until every coordinate has a bulk and tail
#     ESS of n_shard_draws, and samples the reference posterior with Stan;
#  3. runs D&C ABF for every (zeta, N, replica) of sweep_grid;
#  4. runs CMC for every (N, replica), and computes the IAD of every run on
#     (alpha, beta, gamma, omega, u).
# Each ABF replica uses the first N draws of block 1 of every shard; CMC
# replica r uses the first N draws of block r.
#
# Step 2 saves its result in results_dir, and step 3 saves each replica in
# results_dir/run_name. The script skips any result that is already there, so
# it can be stopped and restarted. Change run_name for each variant below, so
# that its runs are kept apart; the Stan draws are shared.
#
# Output: results_dir/run_name/plot_data.rds, with one row per run (method,
# zeta, N, replica, runtime, IAD, error of the mean of omega). Draw the figures
# with plotting/plot_iad_summary.R and plotting/plot_backblaze_pairs.R. The
# script also prints Table tab:ph_zeta_N.
#
# Run time: with 64 cores, about 7 hours for step 2 and 40 minutes for the 120
# ABF runs of step 3 (the paper's runs); longer with fewer cores.
#
# Variants:
#  - Table tab:ph_finegrid: run_name <- "finegrid_k1", zeta_vec <- 0.85,
#    N_vec <- 30000L, nRep <- 5L, use_offline_grid <- TRUE and
#    offline_grid_refine <- 1L; then again with k = 2 and k = 4. Each fusion
#    first runs the adaptive SMC as a pilot, then reruns it on the pilot's grid
#    with each step split into offline_grid_refine parts. The row "Adaptive" of
#    the table is the run zeta = 0.85, N = 30,000 of the default settings.
#  - Smaller runs: reduce zeta_vec, N_vec, nRep, N_max or NMCtruth. The Stan
#    sampling keeps 10 * N_max draws per shard.
#  - Step size: step_size_criterion <- "asymptotic"; tune_nchunks <- TRUE takes
#    the first step from the bisection on the one-step ESS.
#  - Preconditioning: preconditioning <- "no" or "diagonal".

library(this.path)
setwd(this.path::here(.. = 1))
source("R/common_setup.R") # this clears the workspace
library(data.table)
library(rstan)
source("R/stan_shard_sampling.R")

total_start <- Sys.time()

## Settings ##
model_name <- "phbackblaze_centered_grouped_halfnormal"
nShards <- 16L # a power of two
nFamilies <- 4L
nCores <- min(64L, parallel::detectCores()) # the paper used 64
eff_seed <- 20329L # the seed of the paper's runs
data_seed <- 1L # order of the drives within each stratum
shard_seed <- 2L # split of the drives among the shards

run_name <- "sweep"
zeta_vec <- c(0.85, 0.9, 0.95, 0.99) # target one-step ESS
N_vec <- c(1000L, 10000L, 30000L) # particles
nRep <- 10L
sweep_grid <- expand.grid(rep_idx = seq_len(nRep), zeta = zeta_vec, N = N_vec)

target_rejection_acceptance <- 0.01
resamp_thr <- 0.5
step_size_criterion <- "ess_ratio"
step_size_min_factor <- 0.5
step_size_max_factor <- 1.5
step_size_total_min <- 1e-5
tune_nchunks <- FALSE
preconditioning <- "full"
use_offline_grid <- FALSE
offline_grid_refine <- 1L
# The replicate seeds are spaced by the seed block of one tree, which depends
# on n_eval_reps (see run_dc_abf()).
n_eval_reps <- 4L

N_max <- 30000L # draws per block; at least max(N_vec)
n_shard_draws <- 10L * N_max # draws kept per shard: 10 blocks
shard_warmup <- 1000L
shard_draws_per_ess <- 4 # raw draws per unit of ESS in the first Stan attempt
NMCtruth <- 256000L # reference sample
truth_chains <- 64L
truth_warmup <- 1000L
iad_grid_n <- 10000L

results_dir <- "results/backblaze"
data_file <- file.path(results_dir, "hds.rda")
shard_file <- file.path(results_dir, "shard_draws.rds")
rep_file <- function(zeta, N, rep_idx) {
  file.path(results_dir, run_name, sprintf("zeta%s_N%d", zeta, N), sprintf("rep%02d.rds", rep_idx))
}
plot_data_file <- file.path(results_dir, run_name, "plot_data.rds")
dir.create(file.path(results_dir, run_name), showWarnings = FALSE, recursive = TRUE)
stopifnot(N_max >= max(N_vec), nRep <= n_shard_draws %/% N_max)

## 1. Data and shards ##
if (!file.exists(data_file)) {
  download.file(
    "https://raw.githubusercontent.com/boennecd/dynamichazard/3c5964085e3d3c458710f0b5c60355793188b377/data/hds.rda",
    data_file, mode = "wb"
  )
}
if (unname(tools::md5sum(data_file)) != "c2e29a112846c5c9e5ea0aa9062c9ab4") {
  stop(sprintf("The MD5 sum of %s is not that of the data used in the paper.", data_file), call. = FALSE)
}
load(data_file)

# One delayed-entry survival record per drive; keep the largest families.
d <- as.data.table(hds)[, .(
  start = min(tstart),
  stop = max(tstop),
  event = max(fails),
  size_tb = size_tb[1],
  n_fails = max(n_fails)
), .(serial_number, model)]
d <- d[start >= 0 & stop > start & size_tb > 0 & n_fails <= 1]
keep <- d[, .N, model][order(-N)][seq_len(nFamilies), model]
d <- d[model %chin% keep]

# All drives are used; the stratified draw below only fixes their order.
nData <- nrow(d)
stratum_allocation <- d[, .(stratum_n = .N), by = .(model, event)]
stratum_allocation[, n_take := stratum_n]
set.seed(data_seed)
d <- merge(d, stratum_allocation[, .(model, event, n_take)], by = c("model", "event"), sort = FALSE)
d <- d[, .SD[sample.int(.N, first(n_take))], by = .(model, event)]
d[, n_take := NULL]

# Ages in years; standardised log capacity.
d[, `:=`(
  start = start / (24 * 365.25),
  stop = stop / (24 * 365.25),
  group = as.integer(factor(model)),
  size_z = as.numeric(scale(log(size_tb)))
)]
G <- max(d$group)
ndim <- 4L + G

# Divide every family-by-event stratum as evenly as possible among the shards.
set.seed(shard_seed)
d[, stratum_id := .GRP, by = .(model, event)]
stratum_sizes <- d[, .N, by = stratum_id][order(stratum_id)]
shard_order <- sample.int(nShards)
shard_offset <- 0L
for (stratum in seq_len(nrow(stratum_sizes))) {
  rows <- which(d$stratum_id == stratum_sizes$stratum_id[stratum])
  assignments <- shard_order[((shard_offset + seq_along(rows) - 1L) %% nShards) + 1L]
  d[rows, shard := assignments[sample.int(length(rows))]]
  shard_offset <- (shard_offset + length(rows)) %% nShards
}
d[, stratum_id := NULL]
shard_data <- split(d, d$shard, drop = TRUE)
cat(sprintf("%d drives, %d failures; %d to %d drives per shard.\n",
            nrow(d), sum(d$event), min(lengths(lapply(shard_data, `[[`, "event"))),
            max(lengths(lapply(shard_data, `[[`, "event")))))

# Each shard carries 1 / nShards of the prior.
Params <- list(
  lapply(shard_data, `[[`, "start"),
  lapply(shard_data, `[[`, "stop"),
  lapply(shard_data, `[[`, "event"),
  lapply(shard_data, `[[`, "size_z"),
  lapply(shard_data, `[[`, "group"),
  G,
  rep(1 / nShards, nShards)
)

## 2. Stan: shard draws and reference sample ##
if (file.exists(shard_file)) {
  cat(sprintf("Reading the shard draws and the reference sample from %s.\n", shard_file))
  stan_results <- readRDS(shard_file)
} else {
  stan_start <- proc.time()[["elapsed"]]
  stan_code <- "
data {
  int<lower=1> N;
  int<lower=1> G;
  vector<lower=0>[N] start;
  vector<lower=0>[N] stop;
  vector<lower=0, upper=1>[N] event;
  vector[N] size_z;
  array[N] int<lower=1, upper=G> group;
  real<lower=0, upper=1> prior_fraction;
}
parameters {
  real alpha;
  real beta;
  real log_rho;
  real log_omega;
  vector[G] u;
}
model {
  real rho = exp(log_rho);
  vector[N] eta = alpha + beta * size_z + u[group];

  target += prior_fraction * (
    normal_lpdf(alpha | -4, 2) +
    normal_lpdf(beta | 0, 1) +
    normal_lpdf(log_rho | log(1.2), 0.7) +
    log_omega - 0.5 * square(exp(log_omega) / 0.8) +
    normal_lpdf(u | 0, exp(log_omega))
  );

  target += dot_product(event, log_rho + (rho - 1) * log(stop) + eta)
            - dot_product(exp(eta), pow(stop, rho) - pow(start, rho));
}
"
  rstan_options(auto_write = TRUE)
  compiled_model <- stan_model(model_code = stan_code)
  parameter_names <- c("alpha", "beta", "log_rho", "log_omega", sprintf("u[%d]", seq_len(G)))
  stan_control <- list(metric = "dense_e", adapt_delta = 0.99, max_treedepth = 50)
  stan_data <- function(x, prior_fraction) list(
    N = nrow(x), G = G, start = x$start, stop = x$stop, event = x$event,
    size_z = x$size_z, group = x$group, prior_fraction = prior_fraction
  )

  # One chain per shard, run until the bulk and tail ESS of every coordinate
  # reach n_shard_draws, then thinned to n_shard_draws draws.
  shard_initial_retained <- ceiling(shard_draws_per_ess * n_shard_draws)
  shard_results <- sample_stan_shards(
    compiled_model = compiled_model,
    shard_stan_data = lapply(shard_data, stan_data, prior_fraction = 1 / nShards),
    stan_seeds = eff_seed + seq_len(nShards),
    thinning_seeds = eff_seed + 1000L + seq_len(nShards),
    n_cores = min(nShards, nCores),
    parameter_names = parameter_names,
    n_particles = n_shard_draws,
    chains = 1L,
    warmup = shard_warmup,
    initial_retained_per_chain = shard_initial_retained,
    max_retained_per_chain = 8L * shard_initial_retained,
    min_bulk_ess = n_shard_draws,
    min_tail_ess = n_shard_draws,
    enforce_rhat_le_1_01 = FALSE,
    enforce_zero_divergences = FALSE,
    enforce_no_treedepth_hits = FALSE,
    chain_cores = 1L,
    control = stan_control
  )
  initPart <- array(NA_real_, dim = c(n_shard_draws, nShards, ndim))
  for (shard in seq_len(nShards)) initPart[, shard, ] <- shard_results[[shard]]$draws

  cat("Sampling the reference posterior.\n")
  truth_fit <- sampling(
    compiled_model,
    data = stan_data(d, prior_fraction = 1),
    chains = truth_chains,
    iter = truth_warmup + NMCtruth %/% truth_chains,
    warmup = truth_warmup,
    seed = eff_seed + 100L,
    cores = min(truth_chains, nCores),
    control = stan_control
  )
  actualPart <- as.matrix(truth_fit)[, parameter_names, drop = FALSE]

  # Bulk and tail ESS, divergences and maximum-treedepth hits of every sample.
  # The divergence and treedepth checks are off, so they are only reported.
  stan_report <- do.call(rbind, c(
    lapply(seq_len(nShards), function(shard) {
      diagnostics <- shard_results[[shard]]$parameter_diagnostics
      data.frame(sample = sprintf("shard_%d", shard),
                 min_bulk_ess = min(diagnostics$ess_bulk), min_tail_ess = min(diagnostics$ess_tail),
                 divergences = shard_results[[shard]]$divergences,
                 treedepth_hits = shard_results[[shard]]$treedepth_hits)
    }),
    list({
      truth_diagnostics <- posterior::summarise_draws(
        posterior::as_draws_array(as.array(truth_fit, pars = parameter_names)),
        ess_bulk = posterior::ess_bulk, ess_tail = posterior::ess_tail
      )
      truth_sampler <- get_sampler_params(truth_fit, inc_warmup = FALSE)
      data.frame(sample = "reference",
                 min_bulk_ess = min(truth_diagnostics$ess_bulk), min_tail_ess = min(truth_diagnostics$ess_tail),
                 divergences = sum(vapply(truth_sampler, function(x) sum(x[, "divergent__"]), numeric(1L))),
                 treedepth_hits = sum(vapply(truth_sampler, function(x) sum(x[, "treedepth__"] >= stan_control$max_treedepth), numeric(1L))))
    })
  ))
  print(stan_report, row.names = FALSE, digits = 4)

  stan_results <- list(
    eff_seed = eff_seed,
    initPart = initPart,
    actualPart = actualPart,
    stan_report = stan_report,
    stan_seconds = proc.time()[["elapsed"]] - stan_start
  )
  saveRDS(stan_results, shard_file)
  rm(shard_results, truth_fit, initPart, actualPart)
}
if (!identical(dim(stan_results$initPart), c(n_shard_draws, nShards, ndim)) ||
    !identical(stan_results$eff_seed, eff_seed)) {
  stop(sprintf("%s was produced with other settings; delete it to rerun the Stan sampling.", shard_file),
       call. = FALSE)
}

## 3. D&C ABF ##
# Replicate r uses the seed eff_seed + (r - 1) * stride, where stride is the
# width of the seed block of one tree, so no two replicates share a seed. The
# seed does not depend on zeta or N.
rep_seed_stride <- (nShards - 1L) * (n_eval_reps + 2L)
for (task_id in seq_len(nrow(sweep_grid))) {
  rep_idx <- sweep_grid$rep_idx[task_id]
  zeta <- sweep_grid$zeta[task_id]
  N <- sweep_grid$N[task_id]
  rep_path <- rep_file(zeta, N, rep_idx)
  if (file.exists(rep_path)) next

  cat(sprintf("ABF run %d of %d: zeta = %s, N = %d, replicate %d.\n",
              task_id, nrow(sweep_grid), zeta, N, rep_idx))
  rep_seed <- eff_seed + (rep_idx - 1L) * rep_seed_stride
  dc_tree <- run_dc_abf(
    leaf_samples = lapply(seq_len(nShards), function(shard) {
      matrix(stan_results$initPart[seq_len(N), shard, ], nrow = N, ncol = ndim)
    }),
    model_name = model_name,
    params = Params,
    n_particles = N,
    # With tune_nchunks = FALSE this only enters the cap on the number of
    # rejection-sampling proposals; kept as in the paper's runs.
    n_tuning_particles = min(N, 2000L),
    seed = rep_seed,
    n_cores = nCores,
    target_rejection_acceptance = target_rejection_acceptance,
    target_one_step_ess = zeta,
    n_eval_reps = n_eval_reps,
    resamp_thr = resamp_thr,
    preconditioning = preconditioning,
    step_size_min_factor = step_size_min_factor,
    step_size_max_factor = step_size_max_factor,
    step_size_total_min = step_size_total_min,
    step_size_criterion = step_size_criterion,
    use_offline_grid = use_offline_grid,
    offline_grid_refine = offline_grid_refine,
    tune_nchunks = tune_nchunks,
    verbose = FALSE
  )
  # The runtime is the wall time of the whole tree.
  dir.create(dirname(rep_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(list(
    zeta = zeta, N = N, rep_idx = rep_idx, seed = rep_seed,
    out = dc_tree$root$out, normw = dc_tree$root$normw,
    elapsed_seconds = dc_tree$elapsed_seconds
  ), rep_path)
  cat(sprintf("  %.1f seconds, final ESS %.0f.\n", dc_tree$elapsed_seconds, 1 / sum(dc_tree$root$normw^2)))
}

## 4. CMC and the IAD of every run ##
# CMC combines the draws in the sampled coordinates, which are unconstrained.
# Every sample is then mapped to (alpha, beta, rho, omega, u), with
# rho = exp(log_rho) and omega = exp(log_omega), before the IAD.
back_transform <- function(x) {
  x[, 3L] <- exp(x[, 3L])
  x[, 4L] <- exp(x[, 4L])
  x
}
truth_draws <- back_transform(unname(stan_results$actualPart))

posterior_samples <- list()
posterior_weights <- list()
plot_rows <- list()
for (task_id in seq_len(nrow(sweep_grid))) {
  run <- readRDS(rep_file(sweep_grid$zeta[task_id], sweep_grid$N[task_id], sweep_grid$rep_idx[task_id]))
  posterior_samples[[task_id]] <- back_transform(run$out)
  posterior_weights[[task_id]] <- run$normw
  plot_rows[[task_id]] <- data.frame(method = "ABF", zeta = run$zeta, N = run$N,
                                     rep_idx = run$rep_idx, elapsed_seconds = run$elapsed_seconds)
}
# CMC replicate r uses the first N draws of block r; only the aggregation is timed.
for (N in N_vec) {
  for (rep_idx in seq_len(nRep)) {
    cmc_input <- stan_results$initPart[(rep_idx - 1L) * N_max + seq_len(N), , , drop = FALSE]
    cmc_start <- proc.time()[["elapsed"]]
    cmc_out <- consensus_mc(cmc_input)
    row_idx <- length(posterior_samples) + 1L
    posterior_samples[[row_idx]] <- back_transform(cmc_out)
    posterior_weights[row_idx] <- list(NULL)
    plot_rows[[row_idx]] <- data.frame(method = "CMC", zeta = NA_real_, N = N, rep_idx = rep_idx,
                                       elapsed_seconds = proc.time()[["elapsed"]] - cmc_start)
  }
}
plot_data <- do.call(rbind, plot_rows)
plot_data$IAD <- integrated_absolute_distance(
  full_post = truth_draws,
  fusion_post = posterior_samples,
  print_res = FALSE,
  grid_n = iad_grid_n,
  fusion_weights = posterior_weights
)
# Error of the weighted mean of omega, in reference standard deviations.
plot_data$omega_error <- vapply(seq_along(posterior_samples), function(i) {
  w <- if (is.null(posterior_weights[[i]])) NULL else as.numeric(posterior_weights[[i]])
  omega_mean <- if (is.null(w)) mean(posterior_samples[[i]][, 4L]) else sum(w * posterior_samples[[i]][, 4L]) / sum(w)
  (omega_mean - mean(truth_draws[, 4L])) / sd(truth_draws[, 4L])
}, numeric(1L))
saveRDS(plot_data, plot_data_file)

summary_table <- aggregate(
  cbind(IAD, omega_error, elapsed_seconds) ~ method + zeta + N,
  data = transform(plot_data, zeta = ifelse(is.na(zeta), -1, zeta)),
  FUN = median
)
summary_table$zeta[summary_table$zeta == -1] <- NA
cat("\nMedian IAD, error of the mean of omega and runtime (seconds) over the replicas:\n")
print(summary_table, row.names = FALSE, digits = 3)
cat(sprintf("Results saved to %s.\n", plot_data_file))
print(Sys.time() - total_start)
