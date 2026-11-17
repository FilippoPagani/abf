################################################################################
#### Logistic regression on the nycflights13 data: D&C ABF and CMC          ####
################################################################################
# Section "Logistic regression on the nycflights13 data" of the paper: the
# results of Figures fig:nycflights_iad_compare and
# fig:nycflights_iad_compare_cmc.
#
# The 327,346 complete flights are shuffled and split among C = 64 shards. The
# model is the logistic regression of Chan et al. (2023), Section 5.4.2, with
# d = 21 and prior N(0, I), so each shard has the fractional prior N(0, C I).
# The script
#  1. builds the data and the shards;
#  2. samples each shard with Stan until every coordinate has a bulk and tail
#     ESS of n_shard_draws, and samples the reference posterior with Stan;
#  3. runs D&C ABF for every (zeta, N, replica) of sweep_grid;
#  4. runs CMC for every (N, replica), and computes the IAD of every run.
# Each ABF replica uses the first N draws of block 1 of every shard; CMC
# replica r uses the first N draws of block r, so the CMC replicas have
# independent shard draws, as in Chan et al. (2023).
#
# Step 2 saves its result in results_dir, and step 3 saves each replica in
# results_dir/run_name. The script skips any result that is already there, so
# it can be stopped and restarted. Change run_name for each variant below, so
# that its runs are kept apart; the Stan draws are shared.
#
# Output: results_dir/run_name/plot_data.rds, with one row per run (method,
# zeta, N, replica, runtime, IAD). Draw the figures with
# plotting/plot_iad_summary.R.
#
# Run time: with 64 cores, about 5 hours for step 2 and 10 hours for the 180
# ABF runs of step 3 (the paper's runs); longer with fewer cores.
#
# Variants:
#  - Smaller runs: reduce zeta_vec, N_vec, nRep, N_max or NMCtruth. The Stan
#    sampling keeps 10 * N_max draws per shard.
#  - Step size: step_size_criterion <- "asymptotic"; tune_nchunks <- TRUE takes
#    the first step from the bisection on the one-step ESS.
#  - Preconditioning: preconditioning <- "no" or "diagonal".
#  - Fixed grid: use_offline_grid <- TRUE runs each fusion first as an adaptive
#    pilot, then reruns it on the pilot's grid with each step split into
#    offline_grid_refine parts.

library(this.path)
setwd(this.path::here(.. = 1))
source("R/common_setup.R") # this clears the workspace
library(rstan)
source("R/stan_shard_sampling.R")

total_start <- Sys.time()

## Settings ##
model_name <- "logit_wprior_countsxypairs"
nShards <- 64L # a power of two
nCores <- min(64L, parallel::detectCores()) # the paper used 64
eff_seed <- 25471L # the seed of the paper's runs

run_name <- "sweep"
zeta_vec <- c(0.5, 0.6, 0.7, 0.8, 0.9, 0.99) # target one-step ESS
N_vec <- c(1000L, 10000L, 30000L) # particles
nRep <- 10L
sweep_grid <- expand.grid(rep_idx = seq_len(nRep), zeta = zeta_vec, N = N_vec)

target_rejection_acceptance <- 0.01
resamp_thr <- 0.5
step_size_criterion <- "ess_ratio"
step_size_min_factor <- 0.5
step_size_max_factor <- 1.5
step_size_total_min <- 1e-8
tune_nchunks <- FALSE
preconditioning <- "full"
use_offline_grid <- FALSE
offline_grid_refine <- 1L
# The replicate seeds are spaced by the seed block of one tree, which depends
# on n_eval_reps (see run_dc_abf()).
n_eval_reps <- 4L

prior_mean_value <- 0
prior_full_variance <- 1

N_max <- 30000L # draws per block; at least max(N_vec)
n_shard_draws <- 10L * N_max # draws kept per shard: 10 blocks
shard_warmup <- 1000L
NMCtruth <- 256000L # reference sample
truth_chains <- 64L
truth_warmup <- 1000L
iad_grid_n <- 10000L

results_dir <- "results/nycflights13"
shard_file <- file.path(results_dir, "shard_draws.rds")
rep_file <- function(zeta, N, rep_idx) {
  file.path(results_dir, run_name, sprintf("zeta%s_N%d", zeta, N), sprintf("rep%02d.rds", rep_idx))
}
plot_data_file <- file.path(results_dir, run_name, "plot_data.rds")
dir.create(file.path(results_dir, run_name), showWarnings = FALSE, recursive = TRUE)
stopifnot(N_max >= max(N_vec), nRep <= n_shard_draws %/% N_max)

## 1. Data and shards ##
flights_df <- as.data.frame(nycflights13::flights)
nyc_flights <- flights_df[, c("arr_delay", "origin", "carrier", "dep_delay",
                              "year", "day", "month", "hour", "distance")]
nyc_flights <- nyc_flights[complete.cases(nyc_flights), ]

set.seed(eff_seed)
nyc_flights <- nyc_flights[sample.int(nrow(nyc_flights)), ]

nyc_flights$delayed <- as.numeric(nyc_flights$arr_delay > 0)
nyc_flights$dep_delayed <- as.numeric(nyc_flights$dep_delay > 0)
flight_date <- as.Date(sprintf("%04d-%02d-%02d", nyc_flights$year, nyc_flights$month, nyc_flights$day))
nyc_flights$weekend <- as.numeric(as.POSIXlt(flight_date)$wday %in% c(0, 6))
nyc_flights$night <- as.numeric(nyc_flights$hour >= 20 | nyc_flights$hour <= 5)

# One dummy per carrier except the last, and one per origin except LGA.
carrier_names <- names(table(nyc_flights$carrier))
carrier_cols <- carrier_names[seq_len(length(carrier_names) - 1L)]
for (carrier in carrier_cols) {
  nyc_flights[[carrier]] <- as.numeric(nyc_flights$carrier == carrier)
}
nyc_flights$EWR <- as.numeric(nyc_flights$origin == "EWR")
nyc_flights$JFK <- as.numeric(nyc_flights$origin == "JFK")

x_cols <- c("dep_delayed", "weekend", "night", carrier_cols, "EWR", "JFK")
X <- cbind(intercept = 1, as.matrix(nyc_flights[, x_cols, drop = FALSE]))
storage.mode(X) <- "double"
y <- nyc_flights$delayed
ndim <- ncol(X)
stopifnot(ndim == 21L)

# Round-robin split of the shuffled flights.
shard_id <- rep(seq_len(nShards), length.out = nrow(X))
shard_index_list <- split(seq_len(nrow(X)), shard_id)
X_list <- lapply(shard_index_list, function(idx) X[idx, , drop = FALSE])
y_list <- lapply(shard_index_list, function(idx) y[idx])
cat(sprintf("%d complete flights; shard sizes %s.\n", nrow(X),
            paste(range(lengths(y_list)), collapse = " to ")))

# Carriers that do not appear in a shard: there the coefficient is determined
# by the fractional prior alone.
for (shard in seq_len(nShards)) {
  absent <- colnames(X)[-1L][colSums(X_list[[shard]][, -1L, drop = FALSE]) == 0]
  if (length(absent) > 0L) cat(sprintf("Shard %d has no flight with %s.\n", shard, paste(absent, collapse = ", ")))
}

prior_mean <- rep(prior_mean_value, ndim)
prior_full_precision <- rep(1 / prior_full_variance, ndim)

## 2. Stan: shard draws and reference sample ##
if (file.exists(shard_file)) {
  cat(sprintf("Reading the shard draws and the reference sample from %s.\n", shard_file))
  stan_results <- readRDS(shard_file)
} else {
  stan_start <- proc.time()[["elapsed"]]
  stan_code <- "
data {
  int<lower=1> N;
  int<lower=1> D;
  matrix[N, D] X;
  array[N] int<lower=0, upper=1> y;
  vector[D] prior_mean;
  vector<lower=0>[D] prior_precision;
}
parameters {
  vector[D] beta;
}
model {
  target += -0.5 * dot_product(prior_precision, square(beta - prior_mean));
  target += bernoulli_logit_lpmf(y | X * beta);
}
"
  rstan_options(auto_write = TRUE)
  compiled_model <- stan_model(model_code = stan_code)
  parameter_names <- sprintf("beta[%d]", seq_len(ndim))
  stan_control <- list(metric = "dense_e", adapt_delta = 0.99, max_treedepth = 10)

  # One chain per shard, run until the bulk and tail ESS of every coordinate
  # reach n_shard_draws, then thinned to n_shard_draws draws.
  shard_initial_retained <- ceiling(2 * n_shard_draws)
  shard_results <- sample_stan_shards(
    compiled_model = compiled_model,
    shard_stan_data = lapply(seq_len(nShards), function(shard) list(
      N = nrow(X_list[[shard]]), D = ndim, X = X_list[[shard]], y = as.integer(y_list[[shard]]),
      prior_mean = prior_mean, prior_precision = prior_full_precision / nShards
    )),
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
    data = list(N = nrow(X), D = ndim, X = X, y = as.integer(y),
                prior_mean = prior_mean, prior_precision = prior_full_precision),
    chains = truth_chains,
    iter = truth_warmup + NMCtruth %/% truth_chains,
    warmup = truth_warmup,
    seed = eff_seed + 2000L,
    cores = min(truth_chains, nCores),
    control = stan_control
  )
  actualPart <- as.matrix(truth_fit)[, parameter_names, drop = FALSE]

  # Bulk and tail ESS, divergences and maximum-treedepth hits of every sample.
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
# The shard log-likelihoods are sums over the 461 distinct (x, y) rows,
# weighted by the number of flights of the shard with each row.
Params <- c(build_logit_counts_xypairs(X_list, y_list), list(prior_mean, prior_full_precision))

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
posterior_samples <- list()
posterior_weights <- list()
plot_rows <- list()
for (task_id in seq_len(nrow(sweep_grid))) {
  run <- readRDS(rep_file(sweep_grid$zeta[task_id], sweep_grid$N[task_id], sweep_grid$rep_idx[task_id]))
  posterior_samples[[task_id]] <- run$out
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
    posterior_samples[[row_idx]] <- cmc_out
    posterior_weights[row_idx] <- list(NULL)
    plot_rows[[row_idx]] <- data.frame(method = "CMC", zeta = NA_real_, N = N, rep_idx = rep_idx,
                                       elapsed_seconds = proc.time()[["elapsed"]] - cmc_start)
  }
}
plot_data <- do.call(rbind, plot_rows)
plot_data$IAD <- integrated_absolute_distance(
  full_post = stan_results$actualPart,
  fusion_post = posterior_samples,
  print_res = FALSE,
  grid_n = iad_grid_n,
  fusion_weights = posterior_weights
)
saveRDS(plot_data, plot_data_file)

summary_table <- aggregate(
  cbind(IAD, elapsed_seconds) ~ method + zeta + N,
  data = transform(plot_data, zeta = ifelse(is.na(zeta), -1, zeta)),
  FUN = median
)
summary_table$zeta[summary_table$zeta == -1] <- NA
cat("\nMedian IAD and runtime (seconds) over the replicas:\n")
print(summary_table, row.names = FALSE, digits = 3)
cat(sprintf("Results saved to %s.\n", plot_data_file))
print(Sys.time() - total_start)
