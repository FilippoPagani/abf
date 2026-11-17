################################################################################
#### Figure fig:ph_pairs: 2d marginals of the reference, ABF and CMC        ####
################################################################################
# Contour plots of (log omega, u_4), where the centred parameterisation has a
# funnel for the family with the fewest failures, and of (u_3, m_0), where
# m_0 = (log 2 / exp(alpha))^(1 / rho) is the median lifetime in years of a
# drive with x = 0 and u = 0. The rows are the pairs, and the third row shows
# the pair of the second row on zoom_limits. The columns are the Stan
# reference sample, one ABF replica with N = 30,000 resampled according to its
# weights, and the CMC replica with the same index and N. The coordinates are
# taken by position: (alpha, beta, log_rho, log_omega, u_1, ..., u_G). Reads
# the output of experiments/backblaze.R. Adjust the bin sizes of each panel in
# pix_size_grid.

library(this.path)
setwd(this.path::here(.. = 1))
source("R/utilitiesr.R")

options(bitmapType = "cairo")

results_dir <- "results/backblaze"
zeta <- 0.99
N <- 30000L
N_max <- 30000L # draws per block of the shard draws, as in experiments/backblaze.R
rep_idx <- 9L # the replica with the median IAD for zeta = 0.99 and N = 30000
plot_file <- "results/backblaze_pairs_2d_N30k_3rows.pdf"
fig_width <- 8 # inches
# The height of the 8 x 6 figure, plus one row of panels of the same height.
fig_height <- 6 * (1 + 0.26 + 1 + 1) / (1 + 0.26 + 1)
# The axes of the third row, on the pair (u_3, m_0).
zoom_limits <- list(x = c(-1, 2), y = c(0, 100))

stan_results <- readRDS(file.path(results_dir, "shard_draws.rds"))
truth <- unname(stan_results$actualPart)

run <- readRDS(file.path(results_dir, "sweep", sprintf("zeta%s_N%d", zeta, N), sprintf("rep%02d.rds", rep_idx)))
set.seed(run$seed + 7000L)
abf <- run$out[sample.int(nrow(run$out), nrow(run$out), replace = TRUE, prob = run$normw[, 1L]), ]

# As in experiments/backblaze.R, CMC replicate r uses the first N draws of block r.
cmc <- consensus_mc(stan_results$initPart[(rep_idx - 1L) * N_max + seq_len(N), , , drop = FALSE])
rm(stan_results)

conf_levels <- c(.99, seq(.9, .1, by = -.1))
contour_cols <- grDevices::colorRampPalette(
  rev(RColorBrewer::brewer.pal(11, "Spectral"))
)(length(conf_levels))
contour_legend_labels <- paste0(round(100 * conf_levels), "%")

# Histogram pixel size in mm; rows are the pairs, columns reference, ABF, CMC.
pix_size_grid <- matrix(
  c(
    1.2, 1.8, 1.5,
    .35, 0.8, 1.05,
    .7, 1.5, 2.0
  ),
  nrow = 3,
  byrow = TRUE
)

outer_level <- max(conf_levels)

pad_limits <- function(lim, pad) {
  lim + c(-1, 1) * pad * diff(lim)
}

set_up_panels <- function() {
  layout(
    matrix(
      c(
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 10, 10
      ),
      nrow = 4,
      byrow = TRUE
    ),
    heights = c(1, 1, 1, 0.26) # the legend under the panels, as in Figure 3
  )
  # The text sizes of plotting/plot_rosenbrock.R, scaled by
  # fig_width / 10, so that both figures have the same text size once they
  # are scaled to the text width.
  par(
    mar = c(4.5, 4.5, 2.8, 1), # mar = c(bottom, left, top, right)
    cex.axis = 1.75 * fig_width / 10,
    cex.lab = 1.75 * fig_width / 10,
    cex.main = 1.85 * fig_width / 10
  )
}

# Size of one panel in inches, read from a null device with the same layout.
pdf(NULL, width = fig_width, height = fig_height)
set_up_panels()
plot.new()
panel_inches <- par("pin")
dev.off()

# 2d histogram of all the draws of x, with bins of pix_size mm on a panel with
# the given limits, and the counts that enclose each of conf_levels of the
# draws, as in plot_ctour_hist_2d. Unlike there, draws outside the limits are
# counted too, so the contour levels do not depend on the axes. A level is NA,
# and is not drawn, when the fullest bin alone holds more than that share.
hist_2d <- function(x, limits, pix_size) {
  n_bins <- round(panel_inches / (pix_size / 25.4))
  width <- c(diff(limits$x), diff(limits$y)) / n_bins
  breaks <- lapply(1:2, function(j) {
    seq(floor(min(x[, j]) / width[j]) * width[j], (ceiling(max(x[, j]) / width[j]) + 1) * width[j], by = width[j])
  })
  counts <- unclass(table(
    cut(x[, 1L], breaks[[1L]], include.lowest = TRUE),
    cut(x[, 2L], breaks[[2L]], include.lowest = TRUE)
  ))
  sorted <- sort(counts, decreasing = TRUE)
  cumulative <- cumsum(sorted) / sum(sorted)
  list(
    x = (breaks[[1L]][-1L] + breaks[[1L]][-length(breaks[[1L]])]) / 2,
    y = (breaks[[2L]][-1L] + breaks[[2L]][-length(breaks[[2L]])]) / 2,
    z = counts,
    levels = sapply(conf_levels, function(level) {
      if (cumulative[1L] > level) NA else sorted[max(which(cumulative <= level))]
    })
  )
}

# Limits that contain the outer contour of every sample. The bin width depends
# on the limits and the contours on the bin width, so the two are updated in
# turn a few times, starting from the central 99% of each coordinate.
get_pair_limits <- function(samples_list, pix_sizes) {
  limits <- lapply(list(x = 1L, y = 2L), function(j) {
    range(sapply(samples_list, function(x) quantile(x[, j], c(0.005, 0.995))))
  })
  for (iteration in 1:5) {
    lines <- unlist(Map(function(x, pix_size) {
      h <- hist_2d(x, limits, pix_size)
      contourLines(h$x, h$y, h$z, levels = h$levels[1L])
    }, samples_list, pix_sizes), recursive = FALSE)
    limits <- list(
      x = pad_limits(range(unlist(lapply(lines, `[[`, "x"))), pad = 0.05),
      y = pad_limits(range(unlist(lapply(lines, `[[`, "y"))), pad = 0.05)
    )
  }
  limits
}

plot_contour_panel <- function(x, panel_title, limits, pix_size, x_label, y_label) {
  plot(
    x,
    type = "n",
    xlim = limits$x,
    ylim = limits$y,
    xaxs = "i",
    yaxs = "i",
    xlab = x_label,
    ylab = y_label,
    main = panel_title
  )
  h <- hist_2d(x, limits, pix_size)
  for (k in which(!is.na(h$levels))) {
    contour(h$x, h$y, h$z, levels = h$levels[k], add = TRUE, drawlabels = FALSE,
            col = contour_cols[k], method = "flattest")
  }
}

# Columns 1, 3, 4, 7 and 8 are alpha, log_rho, log_omega, u_3 and u_4, with
# rho = exp(log_rho). The first row shows log_omega, in which the funnel is
# easier to see than in omega.
omega_u4 <- function(x) cbind(x[, 4L], x[, 8L])
u3_m0 <- function(x) cbind(x[, 7L], (log(2) / exp(x[, 1L]))^(1 / exp(x[, 3L])))
pair_omega_u4 <- list(truth = omega_u4(truth), abf = omega_u4(abf), cmc = omega_u4(cmc))
pair_u3_m0 <- list(truth = u3_m0(truth), abf = u3_m0(abf), cmc = u3_m0(cmc))
limits_omega_u4 <- get_pair_limits(pair_omega_u4, pix_size_grid[1L, ])
# The long upper tail of m_0 under CMC would squash the other two panels, so
# these limits fit the reference and ABF contours only. The CMC contours are
# still computed from all its draws, and its outer ones run off the panel.
limits_u3_m0 <- get_pair_limits(pair_u3_m0[c("truth", "abf")], pix_size_grid[2L, 1:2])

pdf(file = plot_file, width = fig_width, height = fig_height)

old_par <- par(no.readonly = TRUE)
set_up_panels()

plot_contour_panel(pair_omega_u4$truth, "Reference", limits_omega_u4, pix_size_grid[1L, 1L],
                   expression(log ~ omega), expression(u[4]))
plot_contour_panel(pair_omega_u4$abf, "ABF", limits_omega_u4, pix_size_grid[1L, 2L],
                   expression(log ~ omega), expression(u[4]))
plot_contour_panel(pair_omega_u4$cmc, "CMC", limits_omega_u4, pix_size_grid[1L, 3L],
                   expression(log ~ omega), expression(u[4]))
par(mar = c(4.5, 4.5, 0.8, 1)) # the other rows have no titles, so a smaller top margin
plot_contour_panel(pair_u3_m0$truth, "", limits_u3_m0, pix_size_grid[2L, 1L],
                   expression(u[3]), expression(m[0]))
plot_contour_panel(pair_u3_m0$abf, "", limits_u3_m0, pix_size_grid[2L, 2L],
                   expression(u[3]), expression(m[0]))
plot_contour_panel(pair_u3_m0$cmc, "", limits_u3_m0, pix_size_grid[2L, 3L],
                   expression(u[3]), expression(m[0]))
plot_contour_panel(pair_u3_m0$truth, "", zoom_limits, pix_size_grid[3L, 1L],
                   expression(u[3]), expression(m[0]))
plot_contour_panel(pair_u3_m0$abf, "", zoom_limits, pix_size_grid[3L, 2L],
                   expression(u[3]), expression(m[0]))
plot_contour_panel(pair_u3_m0$cmc, "", zoom_limits, pix_size_grid[3L, 3L],
                   expression(u[3]), expression(m[0]))

par(mar = c(0, 0, 0, 0))
plot.new()
legend(
  "center",
  title = "Contour probability",
  legend = contour_legend_labels,
  col = contour_cols,
  lwd = 3,
  horiz = TRUE,
  bty = "n",
  cex = 1.55 * fig_width / 10,
  seg.len = 1.5,
  x.intersp = 0.7
)

par(old_par)
dev.off()

cat(sprintf("Plot saved to %s.\n", plot_file))
