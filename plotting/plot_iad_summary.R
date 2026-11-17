################################################################################
#### Figures fig:nycflights_iad_compare, fig:nycflights_iad_compare_cmc     ####
#### and fig:ph_iad: median IAD against runtime                             ####
################################################################################
# Reads the output of experiments/nycflights13.R and experiments/backblaze.R
# and draws the three figures with plotting/iad_summary.R.

library(this.path)
setwd(this.path::here(.. = 1))

summary_script <- "plotting/iad_summary.R"
chan_fig7_axes <- TRUE
show_cmc_line <- TRUE

## nycflights13, without and with CMC ##
plot_data <- readRDS("results/nycflights13/sweep/plot_data.rds")

include_cmc <- FALSE
plot_file <- "results/nycflights13_iad_zeta_N_sweep_summary64_nolines.pdf"
source(summary_script)

include_cmc <- TRUE
plot_file <- "results/nycflights13_iad_zeta_N_sweep_summary64_wCMC_nolines.pdf"
source(summary_script)

## Backblaze, ABF at zeta = 0.95 and 0.99 only ##
# The CMC rows are kept for the dashed line at the smallest mean IAD of CMC.
plot_data <- readRDS("results/backblaze/sweep/plot_data.rds")
plot_data <- subset(plot_data, !(method == "ABF" & !zeta %in% c(0.95, 0.99)))

include_cmc <- FALSE
plot_file <- "results/backblaze_iad_zeta_N_sweep_summary_zeta095_099_nolines.pdf"
source(summary_script)
