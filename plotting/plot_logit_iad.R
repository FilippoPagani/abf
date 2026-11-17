################################################################################
#### Figure fig:logistic_iad_compare: IAD against runtime, ABF, BF and CMC  ####
################################################################################
# Reads the output of experiments/logit.R. One point per replica.

library(this.path)
setwd(this.path::here(.. = 1))
library(ggplot2)

options(bitmapType = "cairo")

results_file <- "results/logit/logit.rds"
plot_file <- "results/logit_iad_zeta_sweep_full.pdf"
log_base <- 2

plot_data <- readRDS(results_file)$plot_data

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

x_axis_breaks <- data_log_breaks(plot_data$elapsed_seconds, x_axis_n_breaks)
y_axis_breaks <- data_log_breaks(plot_data$IAD, y_axis_n_breaks)

# Hollow and filled shapes alternate over the zetas in ascending order, so that
# adjacent zetas are easy to tell apart; a cross for CMC.
zeta_shapes <- c(
  "0.1" = 1, "0.2" = 17, "0.3" = 0, "0.4" = 18,
  "0.5" = 2, "0.8" = 16, "0.9" = 5, "0.99" = 15,
  "CMC" = 4
)
plot_data$zeta_label <- factor(
  ifelse(plot_data$method == "CMC", "CMC", as.character(plot_data$zeta)),
  levels = names(zeta_shapes)
)

method_colours <- c(
  "ABF" = "#D55E00",
  "BF" = "#0072B2",
  "CMC" = "#009E73"
)

iad_plot <- ggplot(
  plot_data,
  aes(x = elapsed_seconds, y = IAD, colour = method, shape = zeta_label)
) +
  geom_point(size = 2.1, alpha = 0.7) +
  scale_colour_manual(values = method_colours, name = "Method") +
  scale_shape_manual(values = zeta_shapes, name = expression(zeta)) +
  scale_x_continuous(
    trans = scales::log_trans(base = log_base),
    breaks = x_axis_breaks,
    labels = decimal_labels,
    expand = expansion(mult = c(0.08, 0.12))
  ) +
  scale_y_continuous(
    trans = scales::log_trans(base = log_base),
    breaks = y_axis_breaks,
    labels = scales::label_number(accuracy = 0.001),
    expand = expansion(mult = c(0.08, 0.04))
  ) +
  labs(
    x = "Time elapsed (seconds)",
    y = "Integrated Absolute Distance"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
    legend.position = "right"
  )

if (interactive()) {
  print(iad_plot)
}
ggsave(plot_file, iad_plot, width = 8, height = 5)

cat(sprintf("Plot saved to %s.\n", plot_file))
