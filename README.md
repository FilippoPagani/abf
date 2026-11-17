# Approximate Bayesian Fusion

Code for the paper *Approximate Bayesian Fusion* by Filippo Pagani, Adam M. Johansen and Gareth O. Roberts. It implements Approximate Bayesian Fusion (ABF), which combines samples from the subposteriors of a partitioned dataset into a sample from the full posterior. It also reproduces the experiments of the paper: the four examples of Section 6 and the related appendices.

## Requirements

- R (tested with R 4.5.1) and a C++ compiler with OpenMP. The C++ code is compiled with `Rcpp::sourceCpp()` the first time a script runs.
- R packages: `Rcpp`, `RcppArmadillo`, `this.path`, `ggplot2`, `patchwork`, `scales`, `RColorBrewer`, `IDPmisc`, `RhpcBLASctl` (optional), and
  - `layeredBB` for exact Bayesian Fusion (logistic regression);
  - `rstan`, `posterior` and `nycflights13` for the nycflights13 example;
  - `rstan`, `posterior` and `data.table` for the proportional hazards example. The script downloads the data (the `hds` data set of the R package `dynamichazard`) from a fixed commit and checks its MD5 sum.

The scripts were tested with Rcpp 1.1.1.1.1, RcppArmadillo 14.6.3, rstan 2.32.7, posterior 1.7.0, layeredBB 1.0, data.table 1.18.4, nycflights13 1.0.2, ggplot2 4.0.3 and patchwork 1.3.2.

## Layout

```text
src/model_classes.hpp       the four models of the paper: phi of each shard
src/utilitiesc.cpp          the ABF engine, exact BF, rejection sampler, choice of T,
                            Random-Walk Metropolis
R/common_setup.R            compiles src/ and loads R/; sourced by every experiment
R/utilitiesr.R              preconditioner, bisection on n, logistic-regression data,
                            IAD, CMC, contour plots
R/smc_runner.R              single fusion with replicas (logistic regression)
R/dc_abf_runner.R           divide-and-conquer fusion on a binary tree
R/stan_shard_sampling.R     Stan sampling of the shards up to a target ESS
experiments/                one script per example
plotting/                   one script per figure; saves it in results/
results/                    the output of the experiments and the figures (not tracked by git)
```

## Reproducing the results

Run each script from any directory, for example `Rscript experiments/logit.R`. Each script sets the working directory to the root of the repository. The settings are at the top of each experiment script and are those of the paper. The header of each script describes what the script does and how to run the variants (fixed or adaptive grid, step-size rule, preconditioning, smaller runs).

| Paper | Run | Then plot |
|---|---|---|
| Figure `fig:logistic_iad_compare`, Table `tab:logistic_iad_comparison` | `experiments/logit.R` (prints the table) | `plotting/plot_logit_iad.R` |
| Figure `fig:step_size_compare` | `experiments/logit.R` twice, with the settings in its header | `plotting/plot_logit_step_size_rules.R` |
| Figure `fig:rosen5d_cmc_compare` | `experiments/rosenbrock.R` | `plotting/plot_rosenbrock.R` |
| Figures `fig:nycflights_iad_compare`, `fig:nycflights_iad_compare_cmc` | `experiments/nycflights13.R` | `plotting/plot_iad_summary.R` |
| Figure `fig:ph_iad`, Table `tab:ph_zeta_N` | `experiments/backblaze.R` (prints the table) | `plotting/plot_iad_summary.R` |
| Figure `fig:ph_pairs` | `experiments/backblaze.R` | `plotting/plot_backblaze_pairs.R` |
| Table `tab:ph_finegrid` | `experiments/backblaze.R` three times, with the settings in its header | (printed) |

`plotting/plot_iad_summary.R` needs the results of both `experiments/nycflights13.R` and `experiments/backblaze.R`.

The seeds are fixed to those of the paper's runs. The shard split of each example is therefore the same as in the paper. The Stan draws, and hence the results that depend on them, can still differ slightly across platforms and versions.

### Run times

The nycflights13 and proportional hazards experiments ran on a 64-core machine. `nCores` sets the number of threads of each fusion, and the scripts use at most the number of cores of the machine.

| Script | Run time |
|---|---|
| `experiments/logit.R` | about 2 hours on 10 cores, mostly for exact BF |
| `experiments/rosenbrock.R` | about 25 minutes on 10 cores |
| `experiments/nycflights13.R` | about 5 hours of Stan sampling and 10 hours of ABF on 64 cores |
| `experiments/backblaze.R` | about 7 hours of Stan sampling and 40 minutes of ABF on 64 cores |

The two long scripts save the Stan draws and each ABF replica in `results/` as they go, and skip the results that are already there. A script that is stopped can therefore be restarted.

## Using ABF with another model

A model is a class in `src/model_classes.hpp` that stores the data of the shards and computes phi of each shard for a matrix of particles. It does so with no, diagonal or full preconditioning (`calc_phi_for_all_rows`, `..._diagonal`, `..._full`), together with the sums over the shards that the engine calls. `Rosen5d` is the simplest example. A new model also needs:

- a branch in the dispatcher of `bayesian_fusion_smc` and an overload of `calc_phi_cube_layer_sums_preconditioned` and `calc_phi_fused_last_step_preconditioned` in `src/utilitiesc.cpp`;
- its name in the lists of models in `R/utilitiesr.R`;
- for divide-and-conquer fusion, a rule in `dc_make_node_params()` in `R/dc_abf_runner.R` that builds the data and the fractional prior of a node from those of its leaves.

## Licence

MIT; see `LICENSE`.
