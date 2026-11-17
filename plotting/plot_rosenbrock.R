################################################################################
#### Figure fig:rosen5d_cmc_compare: Rosenbrock 2d marginals                ####
################################################################################
# Contours of the 2d marginals (theta_1, theta_2) and (theta_2, theta_3) of the
# i.i.d. reference sample, of ABF (resampled according to its weights) and of
# CMC. Reads the output of experiments/rosenbrock.R.

library(this.path)
setwd(this.path::here(.. = 1))
source("R/utilitiesr.R")

results_file <- "results/rosenbrock/rosenbrock.rds"
plot_file <- "results/rosen5d_dc_cmc_compare_full.pdf"
res <- readRDS(results_file)

# ABF resampled according to its weights.
set.seed(res$eff_seed + 7000L)
fusion_plot_sample <- res$abf[
  sample.int(
    n = nrow(res$abf),
    size = nrow(res$abf),
    replace = TRUE,
    prob = res$abf_weights
  ),
  ,
  drop = FALSE
]

conf_levels <- c(.99, seq(.9, .1, by = -.1))
contour_cols <- grDevices::colorRampPalette(
  rev(RColorBrewer::brewer.pal(11, "Spectral"))
)(length(conf_levels))
contour_legend_labels <- paste0(round(100 * conf_levels), "%")

# Rows are component pairs (1,2), (2,3); columns are i.i.d. sample, ABF, CMC.
pix_size_grid <- matrix(
  c(
    1.5, 1.5, 1.4,
    .7, .7, 1.0
  ),
  nrow = 2,
  byrow = TRUE
)

outer_level <- max(conf_levels)
tail_prob <- (1 - outer_level) / 2

pad_limits <- function(lim, pad = 0.2) {
  width <- diff(lim)
  lim + c(-1, 1) * pad * width
}

extract_component_pair <- function(x, dims) {
  x[, dims, drop = FALSE]
}

get_pair_limits <- function(samples_list) {
  all_pair_xy <- do.call(rbind, samples_list)

  pair_x_lim <- as.numeric(quantile(
    all_pair_xy[, 1L],
    c(tail_prob, 1 - tail_prob),
    na.rm = TRUE
  ))
  pair_y_lim <- as.numeric(quantile(
    all_pair_xy[, 2L],
    c(tail_prob, 1 - tail_prob),
    na.rm = TRUE
  ))

  list(
    x = pad_limits(pair_x_lim, pad = 0.3),
    y = pad_limits(pair_y_lim, pad = 0.15)
  )
}

plot_contour_panel <- function(x, panel_title, limits, pix_size) {
  plot(
    x,
    type = "n",
    xlim = limits$x,
    ylim = limits$y,
    xaxs = "i",
    yaxs = "i",
    xlab = "",
    ylab = "",
    main = panel_title
  )
  plot_ctour_hist_2d(
    res3 = x,
    pix_size = pix_size,
    conf_levels = conf_levels,
    create_canvas = FALSE
  )
}

reserve_legend_region <- function() {
  old_mar <- par("mar")

  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend_fig <- par("fig")
  par(mar = old_mar)

  legend_fig
}

draw_contour_legend <- function(legend_fig) {
  old_mar <- par("mar")
  old_oma <- par("oma")
  old_xpd <- par("xpd")

  par(fig = legend_fig, mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0),
      new = TRUE, xpd = NA)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")
  legend(
    x = 0.5,
    y = 0.5,
    xjust = 0.5,
    yjust = 0.5,
    title = "Contour probability",
    legend = contour_legend_labels,
    col = contour_cols,
    lwd = 3,
    horiz = TRUE,
    bty = "n",
    cex = 1.55,
    seg.len = 1.5,
    x.intersp = 0.7
  )

  par(mar = old_mar, oma = old_oma, xpd = old_xpd)
}

pair12 <- list(
  truth = extract_component_pair(res$truth, c(1L, 2L)),
  fusion = extract_component_pair(fusion_plot_sample, c(1L, 2L)),
  cmc = extract_component_pair(res$cmc, c(1L, 2L))
)
pair23 <- list(
  truth = extract_component_pair(res$truth, c(2L, 3L)),
  fusion = extract_component_pair(fusion_plot_sample, c(2L, 3L)),
  cmc = extract_component_pair(res$cmc, c(2L, 3L))
)

limits12 <- get_pair_limits(pair12)
limits23 <- get_pair_limits(pair23)

pdf(file = plot_file, width = 10, height = 8)

options(bitmapType = "cairo") ## for transparency

old_par <- par(no.readonly = TRUE)
legend_row_height <- 0.18
layout(
  matrix(
    c(
      1, 2, 3,
      4, 5, 6,
      7, 7, 7
    ),
    nrow = 3,
    byrow = TRUE
  ),
  heights = c(1, 1, legend_row_height)
)
par(
  mar = c(2.8, 4, 2.8, 1), # mar = c(bottom, left, top, right)
  oma = c(0, 0, 2.5, 0),
  cex.axis = 1.75,
  cex.lab = 1.75,
  cex.main = 1.85
)

plot_contour_panel(pair12$truth, "i.i.d. sample", limits12, pix_size_grid[1L, 1L])
plot_contour_panel(pair12$fusion, "ABF", limits12, pix_size_grid[1L, 2L])
plot_contour_panel(pair12$cmc, "CMC", limits12, pix_size_grid[1L, 3L])
par(mar = c(2.8, 4, 0.8, 1)) # the second row has no titles, so a smaller top margin
plot_contour_panel(pair23$truth, "", limits23, pix_size_grid[2L, 1L])
plot_contour_panel(pair23$fusion, "", limits23, pix_size_grid[2L, 2L])
plot_contour_panel(pair23$cmc, "", limits23, pix_size_grid[2L, 3L])
legend_fig <- reserve_legend_region()
mtext("Rosenbrock 5d comparison: components (1,2) and (2,3)", outer = TRUE, cex = 1.35)
draw_contour_legend(legend_fig)

layout(1)
par(old_par)
dev.off()

cat(sprintf("Plot saved to %s.\n", plot_file))
