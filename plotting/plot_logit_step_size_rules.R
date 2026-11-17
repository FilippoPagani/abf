################################################################################
#### Figure fig:step_size_compare: steps chosen by the CESS and the         ####
#### asymptotic rule                                                        ####
################################################################################
# Reads the two runs of experiments/logit.R described in its header, one with
# each rule. Left: after each transition, the step that the rule driving the
# run recommends for the next transition, against the bridge time over T, one
# line per replica, on a broken vertical axis with one band per rule. Right:
# the CESS ratio of each transition, with the target as a dashed line.

library(this.path)
setwd(this.path::here(.. = 1))
library(ggplot2)
library(patchwork)

options(bitmapType = "cairo")

results_files <- c(cess = "results/logit/step_size_cess.rds",
                   asymptotic = "results/logit/step_size_asymptotic.rds")
plot_file <- "results/logit_step_size_rules.pdf"

res <- lapply(results_files, readRDS)
# Both runs share the seed, so they share the coalescence time.
stopifnot(isTRUE(all.equal(res$cess$end_time, res$asymptotic$end_time)))
steps <- do.call(rbind, lapply(names(res), function(rule) cbind(rule = rule, res[[rule]]$steps)))
steps$time_frac <- steps$time / res$cess$end_time
target_one_step_ess <- unique(steps$zeta)

rule_labels <- c(cess = "CESS rule", asymptotic = "Asymptotic rule")
steps$rule_label <- factor(rule_labels[steps$rule], levels = rule_labels)
# Each run takes the step that its own rule recommends.
steps$delta_recommended <- ifelse(steps$rule == "cess", steps$delta_ess, steps$delta_asymptotic)

# Okabe-Ito colours, as in the other figures.
rule_colours <- setNames(c("#D55E00", "#0072B2"), rule_labels)

# A short diagonal mark on the vertical axis, at the bottom (y = 0) or the top
# (y = 1) of a panel, to show where the axis is broken.
break_mark <- function(y) {
  annotation_custom(grid::segmentsGrob(
    x0 = grid::unit(-0.02, "npc"), x1 = grid::unit(0.02, "npc"),
    y0 = grid::unit(y - 0.04, "npc"), y1 = grid::unit(y + 0.04, "npc"),
    gp = grid::gpar(col = "grey30", lwd = 1.5)
  ))
}

# Three vertical dots below the bottom of the upper panel, between the two
# break marks, with a wider bottom margin to make room for them.
break_dots <- function() {
  list(
    annotation_custom(grid::pointsGrob(
      x = grid::unit(rep(0, 3), "npc"),
      y = grid::unit(0, "npc") - grid::unit(c(11, 15.5, 20), "pt"),
      pch = 16, size = grid::unit(2, "pt"),
      gp = grid::gpar(col = "grey30")
    )),
    theme(plot.margin = margin(8, 8, 18, 8))
  )
}

step_band <- function(rule) {
  ggplot(steps[steps$rule == rule, ],
         aes(x = time_frac, y = delta_recommended, colour = rule_label, group = rep_idx)) +
    geom_line(alpha = 0.5) +
    geom_point(size = 1) +
    scale_colour_manual(values = rule_colours, guide = "none") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(n.breaks = 3, labels = function(x) format(x, scientific = FALSE, drop0trailing = TRUE)) +
    coord_cartesian(clip = "off") +
    labs(x = "t / T", y = "Recommended step size") +
    theme_minimal(base_size = 16)
}

# The first step, common to both rules, as a dashed line.
delta_1 <- steps$delta_used[which.min(steps$time)]

step_top <- step_band("cess") +
  geom_hline(yintercept = delta_1, linetype = "dashed", colour = "grey50") +
  scale_y_continuous(breaks = c(0.004, 0.0045, 0.005), labels = c("0.004", "0.0045", "0.005")) +
  coord_cartesian(ylim = c(0.0038, 0.005), clip = "off") +
  break_mark(0) +
  break_dots() +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank(),
        axis.text.y = element_text(colour = "red"))
step_bottom <- step_band("asymptotic") +
  scale_y_continuous(breaks = c(0.00035, 0.0004, 0.00045, 0.0005),
                     labels = c("0.00035", "0.0004", "0.00045", "0.0005")) +
  coord_cartesian(ylim = c(0.00035, 0.0005), clip = "off") +
  break_mark(1)
step_plot <- (step_top / step_bottom) + plot_layout(axis_titles = "collect")

# The first transition is common to both rules, so it is shown only with the
# CESS rule; otherwise the asymptotic lines would cross the axis break.
first_transition <- steps$time == ave(steps$time, steps$rule, steps$rep_idx, FUN = min)
cess_steps <- steps[!(steps$rule == "asymptotic" & first_transition), ]

cess_band <- function(rule) {
  ggplot(cess_steps[cess_steps$rule == rule, ],
         aes(x = time_frac, y = conditional_ess_ratio, colour = rule_label, group = rep_idx)) +
    geom_line(alpha = 0.5, show.legend = TRUE) +
    geom_point(size = 1, show.legend = TRUE) +
    scale_colour_manual(values = rule_colours, drop = FALSE, name = NULL) +
    scale_x_continuous(limits = c(0, 1)) +
    coord_cartesian(clip = "off") +
    labs(x = "t / T", y = "CESS ratio of the transition") +
    theme_minimal(base_size = 16)
}

cess_top <- cess_band("asymptotic") +
  scale_y_continuous(breaks = c(0.98, 0.99, 1)) +
  coord_cartesian(ylim = c(0.978, 1), clip = "off") +
  break_mark(0) +
  break_dots() +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank(),
        axis.text.y = element_text(colour = "red"))
cess_bottom <- cess_band("cess") +
  geom_hline(yintercept = target_one_step_ess, linetype = "dashed", colour = "grey50") +
  scale_y_continuous(breaks = c(0.4, 0.45)) +
  coord_cartesian(ylim = c(0.38, 0.48), clip = "off") +
  break_mark(1)
cess_plot <- (cess_top / cess_bottom) + plot_layout(axis_titles = "collect")

compare_plot <- (step_plot | cess_plot) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

if (interactive()) {
  print(compare_plot)
}
ggsave(plot_file, compare_plot, width = 10, height = 5)
cat(sprintf("Plot saved to %s.\n", plot_file))
