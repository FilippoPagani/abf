############################################################
#### Median IAD against runtime of a D&C ABF sweep        ####
############################################################
# Sourced by plotting/plot_iad_summary.R, with plot_data (the output of
# experiments/nycflights13.R or experiments/backblaze.R) in the workspace. For
# each zeta (or CMC) and number of particles N, a point at the median runtime
# and median IAD over the replicas, with bars from the minimum to the maximum
# of each; colours indicate zeta and shapes N. Settings, read from the
# workspace if set:
#   plot_file       where to save the plot
#   include_cmc     FALSE leaves CMC out of the plot and of the printed summary
#   show_cmc_line   FALSE leaves out the dashed line at the smallest mean IAD
#                   of CMC over N
#   chan_fig7_axes  TRUE draws the log2 axes of Chan et al. (2023), Figure 7

library(ggplot2)

options(bitmapType = "cairo")

# TRUE draws the axes of Chan et al. (2023), Figure 7, on the range of the
# data: log2 of the time in seconds, with the time in seconds on the upper
# edge, and log2 of the IAD, with the IAD on the right edge. FALSE draws
# logarithmic axes labelled in seconds and in IAD.
if (!exists("chan_fig7_axes", inherits = TRUE)) {
  chan_fig7_axes <- FALSE
}
# FALSE leaves CMC out of the plot and of the printed summaries.
if (!exists("include_cmc", inherits = TRUE)) {
  include_cmc <- TRUE
}
# FALSE leaves out the dashed line at the smallest mean IAD of CMC.
if (!exists("show_cmc_line", inherits = TRUE)) {
  show_cmc_line <- TRUE
}
log_base <- 2

if (!exists("plot_data", inherits = TRUE)) {
  stop("plot_data not found: read the results of a sweep before sourcing this script.", call. = FALSE)
}

if (any(!is.finite(plot_data$elapsed_seconds) | plot_data$elapsed_seconds <= 0)) {
  stop("All elapsed times must be positive and finite for the logarithmic x-axis.", call. = FALSE)
}
if (any(!is.finite(plot_data$IAD) | plot_data$IAD <= 0)) {
  stop("All IAD values must be positive and finite for the logarithmic y-axis.", call. = FALSE)
}

# These are target counts. The tick locations are inferred from plot_data.
x_axis_n_breaks <- 9L
y_axis_n_breaks <- 8L

data_log_breaks <- function(values, n) {
  breaks <- scales::breaks_log(n = n, base = 10)(range(values))
  function(limits) breaks[breaks >= min(limits) & breaks <= max(limits)]
}

decimal_labels <- function(x) {
  vapply(
    x,
    function(value) format(
      value,
      scientific = FALSE,
      trim = TRUE,
      big.mark = ",",
      digits = 12
    ),
    character(1)
  )
}

# As in Chan et al. (2023), Figure 7, a dashed line at the smallest mean IAD
# of CMC over N.
cmc_rows <- plot_data$method == "CMC"
cmc_min_mean_IAD <- min(tapply(plot_data$IAD[cmc_rows], plot_data$N[cmc_rows], mean))

# The axes cover the methods drawn and, if drawn, the dashed line.
plotted_rows <- include_cmc | !cmc_rows
axis_seconds <- plot_data$elapsed_seconds[plotted_rows]
axis_IAD <- c(plot_data$IAD[plotted_rows], if (show_cmc_line) cmc_min_mean_IAD)

x_axis_breaks <- data_log_breaks(axis_seconds, x_axis_n_breaks)
y_axis_breaks <- data_log_breaks(axis_IAD, y_axis_n_breaks)

# One level per zeta in ascending order, and one for CMC.
zeta_values <- sort(unique(plot_data$zeta[plot_data$method == "ABF"]))
zeta_levels <- c(as.character(zeta_values), "CMC")
plot_data$zeta_label <- factor(
  ifelse(plot_data$method == "CMC", "CMC", as.character(plot_data$zeta)),
  levels = zeta_levels
)

N_values <- sort(unique(plot_data$N))
plot_data$N_label <- factor(format(plot_data$N, big.mark = ",", trim = TRUE),
                            levels = format(N_values, big.mark = ",", trim = TRUE))

# Median, minimum and maximum of the time and of the IAD over the replicates
# of each zeta and N, ordered by N within each zeta.
summary_data <- do.call(
  rbind,
  lapply(split(plot_data, list(plot_data$zeta_label, plot_data$N), drop = TRUE), function(x) {
    data.frame(
      method = x$method[1L],
      zeta_label = x$zeta_label[1L],
      N = x$N[1L],
      N_label = x$N_label[1L],
      replicates = nrow(x),
      time_median = median(x$elapsed_seconds),
      time_min = min(x$elapsed_seconds),
      time_max = max(x$elapsed_seconds),
      IAD_median = median(x$IAD),
      IAD_min = min(x$IAD),
      IAD_max = max(x$IAD)
    )
  })
)
summary_data <- summary_data[include_cmc | summary_data$method != "CMC", ]
summary_data <- summary_data[order(summary_data$zeta_label, summary_data$N), ]
row.names(summary_data) <- NULL

# Okabe-Ito colours, as in the other figures: warm and cool alternating over
# the zetas in ascending order, and green for CMC as in the logit zeta sweep
# plot.
zeta_colours <- setNames(
  c(c("#E69F00", "#56B4E9", "#D55E00", "#0072B2", "#CC79A7", "#000000")[seq_along(zeta_values)], "#009E73"),
  zeta_levels
)
N_shapes <- setNames(c(16, 17, 15, 18)[seq_along(N_values)], levels(plot_data$N_label))

if (chan_fig7_axes) {
  # The range of the data, rounded outwards to the ticks, which are 2 apart
  # as in Chan et al. (2023), Figure 7.
  log2_elapsed_seconds <- log2(axis_seconds)
  x_limits <- 2 * c(floor(min(log2_elapsed_seconds) / 2), ceiling(max(log2_elapsed_seconds) / 2))
  seconds_breaks <- 10^seq(-3, 7)
  seconds_breaks <- seconds_breaks[log2(seconds_breaks) >= x_limits[1L] &
                                     log2(seconds_breaks) <= x_limits[2L]]
  x_scale <- scale_x_continuous(
    limits = x_limits,
    breaks = seq(x_limits[1L], x_limits[2L], by = 2),
    expand = expansion(mult = 0.02),
    sec.axis = sec_axis(
      ~ 2^.,
      name = "Time elapsed (seconds)",
      breaks = seconds_breaks,
      labels = decimal_labels
    )
  )
  x_label <- "log2(Time elapsed in seconds)"

  # The same for the IAD, with ticks 1 apart.
  log2_IAD <- log2(axis_IAD)
  y_limits <- c(floor(min(log2_IAD)), ceiling(max(log2_IAD)))
  y_scale <- scale_y_continuous(
    limits = y_limits,
    breaks = seq(y_limits[1L], y_limits[2L], by = 1),
    expand = expansion(mult = 0.02),
    sec.axis = sec_axis(
      ~ 2^.,
      name = "IAD",
      # Ticks over the whole axis range, not only the range of the data.
      breaks = data_log_breaks(2^y_limits, 9L),
      labels = scales::label_number(accuracy = 0.01)
    )
  )
  y_label <- "log2(IAD)"

  # The summaries are drawn in log2 units on these axes.
  to_axis <- log2
} else {
  x_scale <- scale_x_continuous(
    trans = scales::log_trans(base = log_base),
    breaks = x_axis_breaks,
    labels = decimal_labels,
    expand = expansion(mult = c(0.08, 0.12))
  )
  x_label <- "Time elapsed (seconds)"

  y_scale <- scale_y_continuous(
    trans = scales::log_trans(base = log_base),
    breaks = y_axis_breaks,
    labels = scales::label_number(accuracy = 0.01),
    expand = expansion(mult = c(0.08, 0.04))
  )
  y_label <- "IAD"

  to_axis <- identity
}

iad_plot <- ggplot(
  summary_data,
  aes(x = to_axis(time_median), y = to_axis(IAD_median), colour = zeta_label, linetype = method)
) +
  (if (show_cmc_line) geom_hline(yintercept = to_axis(cmc_min_mean_IAD), linetype = "dashed", colour = "grey50")) +
  geom_linerange(aes(ymin = to_axis(IAD_min), ymax = to_axis(IAD_max))) +
  geom_linerange(aes(xmin = to_axis(time_min), xmax = to_axis(time_max))) +
  # show.legend = TRUE so that the point is also drawn in the CMC legend.
  geom_point(aes(shape = N_label), size = 3, show.legend = TRUE) +
  # Legends in the order ABF zetas, CMC, N; the N legend only if N varies. The
  # colour legend lists only the zetas, and CMC gets a legend of its own
  # through the linetype, which is solid for both methods.
  scale_colour_manual(
    values = zeta_colours,
    breaks = as.character(zeta_values),
    name = "ABF",
    guide = guide_legend(order = 1)
  ) +
  scale_linetype_manual(
    values = c("ABF" = "solid", "CMC" = "solid"),
    breaks = "CMC",
    name = NULL,
    guide = if (include_cmc) guide_legend(order = 2, override.aes = list(colour = zeta_colours[["CMC"]], shape = 16)) else "none"
  ) +
  scale_shape_manual(
    values = N_shapes,
    name = "N",
    guide = if (length(N_values) > 1) guide_legend(order = 3) else "none"
  ) +
  x_scale +
  y_scale +
  labs(
    x = x_label,
    y = y_label
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5),
    # The time in seconds and the IAD, drawn only if chan_fig7_axes is TRUE,
    # have no grid lines, so they get tick marks.
    axis.ticks.x.top = element_line(colour = "grey30"),
    axis.ticks.y.right = element_line(colour = "grey30"),
    legend.position = "right"
  )

if (interactive()) {
  print(iad_plot)
}
if (exists("plot_file", inherits = TRUE)) {
  ggsave(plot_file, iad_plot, width = 8, height = 5)
}

print(summary_data[, setdiff(names(summary_data), "N_label")], row.names = FALSE, digits = 6)

if (exists("plot_file", inherits = TRUE)) {
  cat(sprintf("Plot saved to %s.\n", plot_file))
}

