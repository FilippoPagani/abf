// The ABF engine and the samplers that feed it, compiled with Rcpp.
//   bayesian_fusion_smc            ABF: the SMC over the coalescing bridges,
//                                  with the adaptive step size, preconditioning
//                                  and an optional fixed time grid
//   bayesian_fusion_smc_lbb_poisson exact BF for the logistic regression, with
//                                  layered Brownian bridges and a Poisson
//                                  estimator of the path integral
//   tune_rejection_end_time_cpp    chooses the coalescence time T for a target
//                                  rejection-sampling acceptance probability
//   rej_samp_init_part             rejection sampler for the initial particles
//   rwLogitChol, rwLogitTuneChol   Random-Walk Metropolis for the logistic
//                                  regression shards and reference sample
// The model classes are in model_classes.hpp.

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "model_classes.hpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]
inline double essw(const arma::vec &weights) {
  const double sum_w = arma::accu(weights);
  const double sum_sq = arma::accu(arma::square(weights));
  return (sum_w * sum_w) / sum_sq;
}


// -------------------------------------------------------------------------
// Numerical helpers
// -------------------------------------------------------------------------

inline int apply_openmp_thread_count(const int n_threads) {
  if (n_threads < 1) {
    Rcpp::stop("n_threads must be a positive integer.");
  }
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
  return n_threads;
#else
  if (n_threads > 1) {
    Rcpp::stop("n_threads > 1 requires compiling utilitiesc.cpp with OpenMP support.");
  }
  return 1;
#endif
}


inline void check_normalized_weights(const arma::vec &weights) {
  // safety checks on weights. Disable for speed.
  
  // catches Infs and NaNs
  bool has_inf = false;
  for (const double& x : weights) {
    if (!std::isfinite(x)) { has_inf = true; break; }
  }
  if ( has_inf ) {
    Rcpp::stop("weights is not finite (is Inf or NaN).");
  }
  if (arma::any(weights < 0.0)) {
    Rcpp::stop("weights must be non-negative.");
  }
  const double sumw = arma::accu(weights);
  if (!(sumw > 0.0)) {
    Rcpp::stop("weights must sum to a positive value.");
  }
  // summation rounding grows with the number of weights, e.g. 1e6 copies of 1/N sum to 1 - 1.3e-11
  const double sum_tol = std::max(1e-12, static_cast<double>(weights.n_elem) * std::numeric_limits<double>::epsilon());
  if ( std::abs(1.0 - sumw) > sum_tol) {
    Rcpp::stop("weights must sum to one.");
  }
}

inline arma::vec calc_weights(const arma::vec &logw) {
  // take the (positive) logw, and calculate the normalised weights.
  const double shift = logw.max();
  const double log_sum_exp = std::log(arma::accu(arma::exp(logw - shift)));
  return arma::exp(logw - shift - log_sum_exp);
}

inline double linear_quantile(const arma::vec &sorted_values, const double probability) {
  const double index = probability * static_cast<double>(sorted_values.n_elem - 1);
  const arma::uword lower = static_cast<arma::uword>(std::floor(index));
  const arma::uword upper = static_cast<arma::uword>(std::ceil(index));
  const double fraction = index - static_cast<double>(lower);
  return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]);
}

inline double weighted_empirical_quantile(const arma::vec &sorted_values,
                                          const arma::vec &sorted_weights,
                                          const double probability) {
  const double target = probability * arma::accu(sorted_weights);
  double cumulative_weight = 0.0;
  for (arma::uword i = 0; i < sorted_values.n_elem; ++i) {
    cumulative_weight += sorted_weights[i];
    if (cumulative_weight >= target) {
      return sorted_values[i];
    }
  }
  return sorted_values[sorted_values.n_elem - 1];
}

inline double abf_round_tolerance(const double scale) {
  return 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

inline double conditional_ess_ratio_unchecked(const arma::vec &weights,
                                              const arma::vec &log_incremental_weights) {
  const double shift = log_incremental_weights.max();
  const arma::vec incremental_weights = arma::exp(log_incremental_weights - shift);
  const double weighted_mean = arma::dot(weights, incremental_weights);
  const double weighted_second_moment = arma::dot(weights, arma::square(incremental_weights));
  return weighted_mean * weighted_mean / weighted_second_moment;
}

struct IncrementalWeightStats {
  double conditional_ess_ratio;
  double conditional_ess;
  double logweight_variance;
  double variance_scale;
  double log_incremental_weight_sd;
  double log_incremental_weight_q01;
  double log_incremental_weight_q50;
  double log_incremental_weight_q99;
  double log_incremental_weight_q99_q01;
};

inline IncrementalWeightStats calculate_incremental_weight_stats(
    const arma::vec &weights,
    const arma::vec &log_incremental_weights,
    const int ordinary_step,
    const double bridge_time) {
  check_normalized_weights(weights);

  IncrementalWeightStats out;
  out.conditional_ess_ratio = conditional_ess_ratio_unchecked(weights, log_incremental_weights);
  if (std::isfinite(out.conditional_ess_ratio) &&
      out.conditional_ess_ratio > 1.0 &&
      out.conditional_ess_ratio - 1.0 <= abf_round_tolerance(1.0)) {
    out.conditional_ess_ratio = 1.0;
  }
  if (!std::isfinite(out.conditional_ess_ratio) ||
      out.conditional_ess_ratio <= 0.0 ||
      out.conditional_ess_ratio > 1.0) {
    Rcpp::stop("conditional ESS ratio is out of range at ordinary step %d and bridge time %.17g.",
               ordinary_step, bridge_time);
  }
  out.conditional_ess = static_cast<double>(weights.n_elem) * out.conditional_ess_ratio;

  const double log_weight_mean = arma::dot(weights, log_incremental_weights);
  const arma::vec centred = log_incremental_weights - log_weight_mean;
  const arma::vec centred_sq = arma::square(centred);
  out.logweight_variance = arma::dot(weights, centred_sq);
  out.variance_scale = centred_sq.max();

  out.log_incremental_weight_sd = arma::stddev(log_incremental_weights, 0);
  const arma::vec sorted_log_weights = arma::sort(log_incremental_weights);
  out.log_incremental_weight_q01 = linear_quantile(sorted_log_weights, 0.01);
  out.log_incremental_weight_q50 = linear_quantile(sorted_log_weights, 0.50);
  out.log_incremental_weight_q99 = linear_quantile(sorted_log_weights, 0.99);
  out.log_incremental_weight_q99_q01 = out.log_incremental_weight_q99 - out.log_incremental_weight_q01;
  return out;
}

inline void record_incremental_weight_diagnostics(
    const arma::vec &weights,
    const arma::vec &log_incremental_weights,
    const int step,
    arma::vec &cessvec,
    arma::vec &log_incremental_weight_sd,
    arma::vec &log_incremental_weight_q01,
    arma::vec &log_incremental_weight_q50,
    arma::vec &log_incremental_weight_q99,
    arma::vec &log_incremental_weight_q99_q01) {
  cessvec[step] = static_cast<double>(weights.n_elem) *
    conditional_ess_ratio_unchecked(weights, log_incremental_weights);

  log_incremental_weight_sd[step] = arma::stddev(log_incremental_weights, 0);
  const arma::vec sorted_log_weights = arma::sort(log_incremental_weights);
  log_incremental_weight_q01[step] = linear_quantile(sorted_log_weights, 0.01);
  log_incremental_weight_q50[step] = linear_quantile(sorted_log_weights, 0.50);
  log_incremental_weight_q99[step] = linear_quantile(sorted_log_weights, 0.99);
  log_incremental_weight_q99_q01[step] = log_incremental_weight_q99[step] - log_incremental_weight_q01[step];
}

inline arma::uvec systematic_residual_resampling(const arma::vec &weights) {
  // in this function, indices start from 0, not from 1 like in R.
  check_normalized_weights(weights);

  const int N = static_cast<int>(weights.n_elem);
  arma::uvec out(N); // will hold the resampled ancestors indices

  // scaled weights sum to N. Scaled is the expected number of offspring for each part.
  const arma::vec scaled = static_cast<double>(N) * weights;
  arma::ivec counts_det(N); // will hold the deterministic part, floor(N w_i)

  int filled = 0;
  for (int i = 0; i < N; ++i) {
    counts_det[i] = static_cast<int>(std::floor(scaled[i])); // copy each particle floor(N w_i) times, deteministic.
    for (int k = 0; k < counts_det[i]; ++k) { // fill out with copies of the unsigned index i
      out[filled++] = static_cast<arma::uword>(i);
    }
  }

  const int n_residual = N - filled; // this is the bit beyond the integer for each particle. It will be assigned at random
  if (n_residual == 0) {
    return out;
  }

  const arma::vec frac = scaled - arma::conv_to<arma::vec>::from(counts_det); // fractional part of N w_i, N x 1 vec
  const double sum_frac = arma::accu(frac);

  const arma::vec cdf = arma::cumsum(frac / sum_frac); //create a cdf
  const double offset = R::runif(0.0, 1.0); // sample one single random offset

  // do the systematic residual draw.
  int j = 0;
  for (int i = 0; i < n_residual; ++i) {
    const double pos = (offset + static_cast<double>(i)) / static_cast<double>(n_residual); // current position
    while ( (j < N - 1) && pos > cdf[j]) {
      ++j; // count how many times pos falls in the current cdf bin, and selects the appropriate particle index
    }
    out[filled++] = static_cast<arma::uword>(j);
  }

  if (filled != N) {
    Rcpp::stop("Internal error in systematic_residual_resampling.");
  }
  return out;
}

// -------------------------------------------------------------------------
// Core particle-system helpers
// -------------------------------------------------------------------------

inline void compute_shard_means(const arma::cube &particles, arma::mat &means) {
  const int ndim = static_cast<int>(particles.n_slices);
  for (int d = 0; d < ndim; ++d) {
    means.col(d) = arma::mean(particles.slice(d), 1);
  }
}

enum class AbfPreconditioningMode {
  No,
  Diagonal,
  Full
};

struct AbfPreconditioner {
  AbfPreconditioningMode mode = AbfPreconditioningMode::No;
  int nshards = 0;
  int ndim = 0;
  arma::mat q_diag;
  arma::mat q_inv_diag;
  arma::mat q_sqrt_diag;
  arma::vec q_star_diag;
  arma::vec q_star_sqrt_diag;
  std::vector<arma::mat> q_cov;
  std::vector<arma::mat> q_inv;
  std::vector<arma::mat> q_chol;
  arma::mat q_star;
  arma::mat q_star_chol;
};

inline AbfPreconditioningMode parse_abf_preconditioning_mode(const std::string &preconditioning) {
  if (preconditioning == "no") {
    return AbfPreconditioningMode::No;
  }
  if (preconditioning == "diagonal") {
    return AbfPreconditioningMode::Diagonal;
  }
  if (preconditioning == "full") {
    return AbfPreconditioningMode::Full;
  }
  Rcpp::stop("preconditioning must be one of 'no', 'diagonal', or 'full'.");
}

inline bool abf_is_finite_mat(const arma::mat &x) {
  const double *ptr = x.memptr();
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    if (!std::isfinite(ptr[i])) {
      return false;
    }
  }
  return true;
}

inline bool abf_is_finite_vec(const arma::vec &x) {
  const double *ptr = x.memptr();
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    if (!std::isfinite(ptr[i])) {
      return false;
    }
  }
  return true;
}

inline void require_abf_list_element(const Rcpp::List &x, const char *name) {
  if (!x.containsElementNamed(name)) {
    Rcpp::stop(std::string("abf_preconditioner is missing '") + name + "'.");
  }
}

inline void check_abf_mat_shape(const arma::mat &x,
                                const int nrow,
                                const int ncol,
                                const std::string &name) {
  if (static_cast<int>(x.n_rows) != nrow || static_cast<int>(x.n_cols) != ncol) {
    Rcpp::stop(name + " has incompatible dimensions.");
  }
  if (!abf_is_finite_mat(x)) {
    Rcpp::stop(name + " must contain only finite values.");
  }
}

inline void check_abf_vec_length(const arma::vec &x,
                                 const int length,
                                 const std::string &name) {
  if (static_cast<int>(x.n_elem) != length) {
    Rcpp::stop(name + " has incompatible length.");
  }
  if (!abf_is_finite_vec(x)) {
    Rcpp::stop(name + " must contain only finite values.");
  }
}

inline void check_abf_positive_matrix_entries(const arma::mat &x,
                                              const std::string &name) {
  const double *ptr = x.memptr();
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    if (!(ptr[i] > 0.0)) {
      Rcpp::stop(name + " must contain positive values.");
    }
  }
}

inline void check_abf_positive_diagonal(const arma::mat &x,
                                        const std::string &name) {
  const int n = static_cast<int>(std::min(x.n_rows, x.n_cols));
  for (int i = 0; i < n; ++i) {
    if (!(x(i, i) > 0.0)) {
      Rcpp::stop(name + " must have positive diagonal values.");
    }
  }
}

inline AbfPreconditioner make_abf_preconditioner(const std::string &preconditioning,
                                                 SEXP abf_preconditioner,
                                                 const int expected_nshards,
                                                 const int expected_ndim) {
  AbfPreconditioner out;
  out.mode = parse_abf_preconditioning_mode(preconditioning);
  out.nshards = expected_nshards;
  out.ndim = expected_ndim;

  if (out.mode == AbfPreconditioningMode::No) {
    return out;
  }

  if (Rf_isNull(abf_preconditioner)) {
    Rcpp::stop("abf_preconditioner must be supplied for empirical preconditioning.");
  }
  if (!Rf_isNewList(abf_preconditioner)) {
    Rcpp::stop("abf_preconditioner must be a list.");
  }

  const Rcpp::List precond(abf_preconditioner);
  require_abf_list_element(precond, "mode");
  require_abf_list_element(precond, "nshards");
  require_abf_list_element(precond, "ndim");

  const std::string list_mode = Rcpp::as<std::string>(precond["mode"]);
  if (list_mode != preconditioning) {
    Rcpp::stop("abf_preconditioner mode does not match preconditioning.");
  }
  const int list_nshards = Rcpp::as<int>(precond["nshards"]);
  const int list_ndim = Rcpp::as<int>(precond["ndim"]);
  if (list_nshards != expected_nshards || list_ndim != expected_ndim) {
    Rcpp::stop("abf_preconditioner dimensions do not match the particle cube.");
  }

  if (out.mode == AbfPreconditioningMode::Diagonal) {
    require_abf_list_element(precond, "q_diag");
    require_abf_list_element(precond, "q_inv_diag");
    require_abf_list_element(precond, "q_sqrt_diag");
    require_abf_list_element(precond, "q_star_diag");
    require_abf_list_element(precond, "q_star_sqrt_diag");

    out.q_diag = Rcpp::as<arma::mat>(precond["q_diag"]);
    out.q_inv_diag = Rcpp::as<arma::mat>(precond["q_inv_diag"]);
    out.q_sqrt_diag = Rcpp::as<arma::mat>(precond["q_sqrt_diag"]);
    out.q_star_diag = Rcpp::as<arma::vec>(precond["q_star_diag"]);
    out.q_star_sqrt_diag = Rcpp::as<arma::vec>(precond["q_star_sqrt_diag"]);

    check_abf_mat_shape(out.q_diag, expected_nshards, expected_ndim, "q_diag");
    check_abf_mat_shape(out.q_inv_diag, expected_nshards, expected_ndim, "q_inv_diag");
    check_abf_mat_shape(out.q_sqrt_diag, expected_nshards, expected_ndim, "q_sqrt_diag");
    check_abf_vec_length(out.q_star_diag, expected_ndim, "q_star_diag");
    check_abf_vec_length(out.q_star_sqrt_diag, expected_ndim, "q_star_sqrt_diag");
    check_abf_positive_matrix_entries(out.q_diag, "q_diag");
    check_abf_positive_matrix_entries(out.q_inv_diag, "q_inv_diag");
    check_abf_positive_matrix_entries(out.q_sqrt_diag, "q_sqrt_diag");
    for (int d = 0; d < expected_ndim; ++d) {
      if (!(out.q_star_diag[d] > 0.0) || !(out.q_star_sqrt_diag[d] > 0.0)) {
        Rcpp::stop("q_star_diag and q_star_sqrt_diag must contain positive values.");
      }
    }
    return out;
  }

  require_abf_list_element(precond, "q_cov");
  require_abf_list_element(precond, "q_inv");
  require_abf_list_element(precond, "q_chol");
  require_abf_list_element(precond, "q_star");
  require_abf_list_element(precond, "q_star_chol");

  const Rcpp::List q_cov_list = Rcpp::as<Rcpp::List>(precond["q_cov"]);
  const Rcpp::List q_inv_list = Rcpp::as<Rcpp::List>(precond["q_inv"]);
  const Rcpp::List q_chol_list = Rcpp::as<Rcpp::List>(precond["q_chol"]);
  if (q_cov_list.size() != expected_nshards ||
      q_inv_list.size() != expected_nshards ||
      q_chol_list.size() != expected_nshards) {
    Rcpp::stop("q_cov, q_inv, and q_chol must have one matrix per shard.");
  }

  out.q_cov.reserve(expected_nshards);
  out.q_inv.reserve(expected_nshards);
  out.q_chol.reserve(expected_nshards);
  for (int c = 0; c < expected_nshards; ++c) {
    arma::mat q_cov_c = Rcpp::as<arma::mat>(q_cov_list[c]);
    arma::mat q_inv_c = Rcpp::as<arma::mat>(q_inv_list[c]);
    arma::mat q_chol_c = Rcpp::as<arma::mat>(q_chol_list[c]);
    check_abf_mat_shape(q_cov_c, expected_ndim, expected_ndim, "q_cov");
    check_abf_mat_shape(q_inv_c, expected_ndim, expected_ndim, "q_inv");
    check_abf_mat_shape(q_chol_c, expected_ndim, expected_ndim, "q_chol");
    check_abf_positive_diagonal(q_cov_c, "q_cov");
    out.q_cov.push_back(q_cov_c);
    out.q_inv.push_back(q_inv_c);
    out.q_chol.push_back(q_chol_c);
  }

  out.q_star = Rcpp::as<arma::mat>(precond["q_star"]);
  out.q_star_chol = Rcpp::as<arma::mat>(precond["q_star_chol"]);
  check_abf_mat_shape(out.q_star, expected_ndim, expected_ndim, "q_star");
  check_abf_mat_shape(out.q_star_chol, expected_ndim, expected_ndim, "q_star_chol");
  check_abf_positive_diagonal(out.q_star, "q_star");

  return out;
}

inline arma::mat get_shard_theta_rows(const arma::cube &particles,
                                      const int shard) {
  const int npart = static_cast<int>(particles.n_rows);
  const int ndim = static_cast<int>(particles.n_slices);
  arma::mat theta_rows(npart, ndim, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    theta_rows.col(d) = particles.slice(d).col(shard);
  }
  return theta_rows;
}

inline void set_shard_theta_rows(arma::cube &particles,
                                 const int shard,
                                 const arma::mat &theta_rows) {
  const int ndim = static_cast<int>(particles.n_slices);
  for (int d = 0; d < ndim; ++d) {
    particles.slice(d).col(shard) = theta_rows.col(d);
  }
}

inline void compute_weighted_centres_diagonal(const arma::cube &particles,
                                              arma::mat &centres,
                                              const AbfPreconditioner &precond) {
  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  centres.zeros(npart, ndim);

  for (int d = 0; d < ndim; ++d) {
    arma::vec weighted_sum(npart, arma::fill::zeros);
    const arma::mat &slice = particles.slice(d);
    for (int c = 0; c < nshards; ++c) {
      weighted_sum += precond.q_inv_diag(c, d) * slice.col(c);
    }
    centres.col(d) = precond.q_star_diag[d] * weighted_sum;
  }
}

inline void compute_weighted_centres_full(const arma::cube &particles,
                                          arma::mat &centres,
                                          const AbfPreconditioner &precond) {
  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  centres.zeros(npart, ndim);

  for (int c = 0; c < nshards; ++c) {
    const arma::mat theta_rows = get_shard_theta_rows(particles, c);
    centres += theta_rows * precond.q_inv[c];
  }
  centres = centres * precond.q_star;
}

inline void update_part_after_resamp(const arma::cube &particles,
                                     const arma::uvec &idx_vec,
                                     arma::cube &out) {
  // assign to particles after resampling the indices according to idx_vec
  const int npart = static_cast<int>(idx_vec.n_elem);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);

  if (static_cast<int>(out.n_rows) != npart ||
      static_cast<int>(out.n_cols) != nshards ||
      static_cast<int>(out.n_slices) != ndim) {
    Rcpp::stop("update_part_after_resamp: arma::cube out incorrect dimensions. ");
  }

  for (int d = 0; d < ndim; ++d) {
    const arma::mat &slice = particles.slice(d); // take slices
    arma::mat &out_slice = out.slice(d);
    for (int c = 0; c < nshards; ++c) {
      const double *src = slice.colptr(c); // pointer start of col c of input (source)
      double *dst = out_slice.colptr(c); // pointer start of col c of output (destination)
      for (int i = 0; i < npart; ++i) {
        dst[i] = src[idx_vec[i]]; // copy rows according to idx_vec (for each d, and c)
      } // columns are contiguous in memory, better than out.row(i) = particles.row(idx_vec[i]);
    }
  }
}

inline void update_phi_reorder_rows(const arma::vec &x,
                                    const arma::uvec &idx,
                                    arma::vec &out) {
  // update phi matrix of current location after resampling by reordering rows (already computed)
  if (out.n_elem != idx.n_elem) {
    Rcpp::stop("out.n_elem != idx.n_elem");
  }
  for (arma::uword i = 0; i < idx.n_elem; ++i) {
    out[i] = x[idx[i]];
  }
}

inline void propagate_bridge(const arma::cube &x_old,
                             arma::cube &x_new,
                             arma::mat &x_bar,
                             arma::mat &psi,
                             arma::vec &chi,
                             arma::vec &shift,
                             const double alpha,
                             const double beta,
                             const double delta_t) {
  // take the bridge one step forward on the time partition
  const int npart = static_cast<int>(x_old.n_rows);
  const int nshards = static_cast<int>(x_old.n_cols);
  const int ndim = static_cast<int>(x_old.n_slices);

  const double sd_psi = std::sqrt(delta_t * alpha); // shard-specific noise
  const double sd_chi = std::sqrt(delta_t * beta / static_cast<double>(nshards)); // common noise

  // some array-size checks
  if (static_cast<int>(psi.n_rows) != npart || static_cast<int>(psi.n_cols) != nshards) {
    Rcpp::stop("psi.n_rows != npart");
  }
  if (static_cast<int>(chi.n_elem) != npart) {
    Rcpp::stop("chi.n_elem) != npart");
  }
  if (static_cast<int>(shift.n_elem) != npart) {
    Rcpp::stop("shift.n_elem != npart");
  }

  compute_shard_means(x_old, x_bar);

  for (int d = 0; d < ndim; ++d) {
    psi.randn();
    chi.randn();

    x_new.slice(d) = alpha * x_old.slice(d) + sd_psi * psi;

    shift = beta * x_bar.col(d) + sd_chi * chi; // same for all the shards
    x_new.slice(d).each_col() += shift;
  }
}

inline void propagate_bridge_diagonal(const arma::cube &x_old,
                                      arma::cube &x_new,
                                      arma::mat &x_bar,
                                      const AbfPreconditioner &precond,
                                      const double alpha,
                                      const double beta,
                                      const double delta_t) {
  const int npart = static_cast<int>(x_old.n_rows);
  const int nshards = static_cast<int>(x_old.n_cols);
  const int ndim = static_cast<int>(x_old.n_slices);
  const double sd_psi = std::sqrt(delta_t * alpha);
  const double sd_chi = std::sqrt(delta_t * beta);

  compute_weighted_centres_diagonal(x_old, x_bar, precond);

  arma::mat psi(npart, nshards, arma::fill::zeros);
  arma::vec chi(npart, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    psi.randn();
    chi.randn();
    const arma::mat &old_slice = x_old.slice(d);
    arma::mat &new_slice = x_new.slice(d);
    const arma::vec common_shift = beta * x_bar.col(d) + sd_chi * precond.q_star_sqrt_diag[d] * chi;
    for (int c = 0; c < nshards; ++c) {
      new_slice.col(c) = alpha * old_slice.col(c) +
        sd_psi * precond.q_sqrt_diag(c, d) * psi.col(c) +
        common_shift;
    }
  }
}

inline void propagate_bridge_full(const arma::cube &x_old,
                                  arma::cube &x_new,
                                  arma::mat &x_bar,
                                  const AbfPreconditioner &precond,
                                  const double alpha,
                                  const double beta,
                                  const double delta_t) {
  const int npart = static_cast<int>(x_old.n_rows);
  const int nshards = static_cast<int>(x_old.n_cols);
  const int ndim = static_cast<int>(x_old.n_slices);
  const double sd_psi = std::sqrt(delta_t * alpha);
  const double sd_chi = std::sqrt(delta_t * beta);

  compute_weighted_centres_full(x_old, x_bar, precond);

  std::vector<arma::mat> shard_z;
  shard_z.reserve(nshards);
  for (int c = 0; c < nshards; ++c) {
    shard_z.emplace_back(npart, ndim, arma::fill::zeros);
  }
  arma::mat common_z(npart, ndim, arma::fill::zeros);
  arma::mat psi(npart, nshards, arma::fill::zeros);
  arma::vec chi(npart, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    psi.randn();
    chi.randn();
    for (int c = 0; c < nshards; ++c) {
      shard_z[c].col(d) = psi.col(c);
    }
    common_z.col(d) = chi;
  }

  const arma::mat common_shift = beta * x_bar + sd_chi * common_z * precond.q_star_chol.t();
  for (int c = 0; c < nshards; ++c) {
    const arma::mat theta_rows = get_shard_theta_rows(x_old, c);
    const arma::mat shard_noise = sd_psi * shard_z[c] * precond.q_chol[c].t();
    const arma::mat theta_new = alpha * theta_rows + common_shift + shard_noise;
    set_shard_theta_rows(x_new, c, theta_new);
  }
}

inline void sample_fused_endpoint(const arma::cube &x_old,
                                  arma::mat &x_bar,
                                  arma::mat &out,
                                  const double delta_t) {
  // this function is only called once on the last step
  const int npart = static_cast<int>(x_old.n_rows);
  const int nshards = static_cast<int>(x_old.n_cols);
  const int ndim = static_cast<int>(x_old.n_slices);

  const double sd_last = std::sqrt(delta_t / static_cast<double>(nshards)); // stdev of the final gaussian draw
  arma::vec z(npart); // temp vec of gaussian noises, one per particle. Not bad, as function only called once

  compute_shard_means(x_old, x_bar);
  for (int d = 0; d < ndim; ++d) { // loop on dimensions
    z.randn(); // sample the noise
    out.col(d) = x_bar.col(d) + sd_last * z; // store the update in out.
  }
}

inline void sample_fused_endpoint_diagonal(const arma::cube &x_old,
                                           arma::mat &x_bar,
                                           arma::mat &out,
                                           const AbfPreconditioner &precond,
                                           const double delta_t) {
  const int npart = static_cast<int>(x_old.n_rows);
  const int ndim = static_cast<int>(x_old.n_slices);
  const double sd_last = std::sqrt(delta_t);
  arma::vec z(npart, arma::fill::zeros);

  compute_weighted_centres_diagonal(x_old, x_bar, precond);
  for (int d = 0; d < ndim; ++d) {
    z.randn();
    out.col(d) = x_bar.col(d) + sd_last * precond.q_star_sqrt_diag[d] * z;
  }
}

inline void sample_fused_endpoint_full(const arma::cube &x_old,
                                       arma::mat &x_bar,
                                       arma::mat &out,
                                       const AbfPreconditioner &precond,
                                       const double delta_t) {
  const int npart = static_cast<int>(x_old.n_rows);
  const int ndim = static_cast<int>(x_old.n_slices);

  compute_weighted_centres_full(x_old, x_bar, precond);

  arma::mat z(npart, ndim, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    z.col(d).randn();
  }
  out = x_bar + std::sqrt(delta_t) * z * precond.q_star_chol.t();
}

inline arma::vec initial_logweights(const arma::cube &particles,
                                    const arma::mat &means,
                                    const double end_time) {
  // if no rejection sampling is done, this function calculates 
  // the initial weights of the particles.
  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  const double inv_two_end_time = 0.5 / end_time;

  arma::vec out(npart, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    const arma::mat &slice = particles.slice(d);
    for (int c = 0; c < nshards; ++c) {
      const double *col = slice.colptr(c);
      for (int i = 0; i < npart; ++i) {
        const double diff_shard = col[i] - means(i, d);
        out[i] += inv_two_end_time * diff_shard * diff_shard;
      }
    }
  }
  return out;
}

inline arma::vec initial_logweights_diagonal(const arma::cube &particles,
                                             const arma::mat &centres,
                                             const AbfPreconditioner &precond,
                                             const double end_time) {
  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  const double inv_two_end_time = 0.5 / end_time;

  arma::vec out(npart, arma::fill::zeros);
  for (int d = 0; d < ndim; ++d) {
    const arma::mat &slice = particles.slice(d);
    for (int c = 0; c < nshards; ++c) {
      const double q_inv = precond.q_inv_diag(c, d);
      const double *col = slice.colptr(c);
      for (int i = 0; i < npart; ++i) {
        const double diff = col[i] - centres(i, d);
        out[i] += inv_two_end_time * q_inv * diff * diff;
      }
    }
  }
  return out;
}

inline arma::vec initial_logweights_full(const arma::cube &particles,
                                         const arma::mat &centres,
                                         const AbfPreconditioner &precond,
                                         const double end_time) {
  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  const double inv_two_end_time = 0.5 / end_time;

  arma::vec out(npart, arma::fill::zeros);
  arma::rowvec diff(ndim, arma::fill::zeros);
  for (int i = 0; i < npart; ++i) {
    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        diff[d] = particles(i, c, d) - centres(i, d);
      }
      out[i] += inv_two_end_time * arma::as_scalar(diff * precond.q_inv[c] * diff.t());
    }
  }
  return out;
}

inline void set_initial_weights(const arma::cube &particles,
                                const arma::mat &means,
                                const double end_time,
                                SEXP init_weight,
                                arma::vec &nlogw,
                                arma::vec &normw,
                                const AbfPreconditioner *precond = nullptr) {
  // this function decided whether to calculate initial weights or not
  // seems slightly redundant, can probably eliminated changing the interface.
  const bool is_string = TYPEOF(init_weight) == STRSXP;

  if (is_string) {
    const std::string init_weight_flag = Rcpp::as<std::string>(init_weight);
    if (init_weight_flag == "y") {
      if (precond == nullptr || precond->mode == AbfPreconditioningMode::No) {
        nlogw = initial_logweights(particles, means, end_time);
      } else if (precond->mode == AbfPreconditioningMode::Diagonal) {
        nlogw = initial_logweights_diagonal(particles, means, *precond, end_time);
      } else {
        nlogw = initial_logweights_full(particles, means, *precond, end_time);
      }
      normw = calc_weights(-nlogw);
      return;
    }
    if (init_weight_flag == "n") {
      normw.fill(1.0 / static_cast<double>(particles.n_rows));
      nlogw = -arma::log(normw);
      return;
    }
  }

  if (Rf_isReal(init_weight)) {
    Rcpp::NumericVector init_w(init_weight);
    if (static_cast<arma::uword>(init_w.size()) != particles.n_rows) {
      Rcpp::stop("init_weight length must equal number of particles.");
    }
    for (double w : init_w) {
      if (!std::isfinite(w) || w <= 0.0) {
        Rcpp::stop("init_weight must contain finite positive values.");
      }
    }
    const double sumw = std::accumulate(init_w.begin(), init_w.end(), 0.0);
    if (std::abs(sumw - 1.0) > 1e-10) {
      Rcpp::stop("init_weight must be normalised.");
    }
    normw = Rcpp::as<arma::vec>(init_w);
    nlogw = -arma::log(normw);
    return;
  }

  Rcpp::stop("weird value for init_weight");
}


inline void calc_phi_cube_layer_sums_preconditioned(const Rosen5d &model,
                                                    const arma::cube &particles,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_cube_layer_sums(particles, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_cube_layer_sums_diagonal(particles, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_cube_layer_sums_full(particles, precond.q_cov, out, gradient_component, hessian_component);
  }
}

inline void calc_phi_cube_layer_sums_preconditioned(const Logit &model,
                                                    const arma::cube &particles,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_cube_layer_sums(particles, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_cube_layer_sums_diagonal(particles, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_cube_layer_sums_full(particles, precond.q_cov, out, gradient_component, hessian_component);
  }
}


inline void calc_phi_cube_layer_sums_preconditioned(const Logit_wprior_countsxypairs &model,
                                                    const arma::cube &particles,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_cube_layer_sums(particles, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_cube_layer_sums_diagonal(particles, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_cube_layer_sums_full(particles, precond.q_cov, out, gradient_component, hessian_component);
  }
}


inline void calc_phi_cube_layer_sums_preconditioned(const PHBackblaze_centered_grouped_halfnormal &model,
                                                    const arma::cube &particles,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_cube_layer_sums(particles, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_cube_layer_sums_diagonal(particles, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_cube_layer_sums_full(particles, precond.q_cov, out, gradient_component, hessian_component);
  }
}


inline void calc_phi_fused_last_step_preconditioned(const Rosen5d &model,
                                                    const arma::mat &fused,
                                                    const int nshards,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_fused_last_step(fused, nshards, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_fused_last_step_diagonal(fused, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_fused_last_step_full(fused, precond.q_cov, out, gradient_component, hessian_component);
  }
}

inline void calc_phi_fused_last_step_preconditioned(const Logit &model,
                                                    const arma::mat &fused,
                                                    const int nshards,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_fused_last_step(fused, nshards, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_fused_last_step_diagonal(fused, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_fused_last_step_full(fused, precond.q_cov, out, gradient_component, hessian_component);
  }
}


inline void calc_phi_fused_last_step_preconditioned(const Logit_wprior_countsxypairs &model,
                                                    const arma::mat &fused,
                                                    const int nshards,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_fused_last_step(fused, nshards, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_fused_last_step_diagonal(fused, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_fused_last_step_full(fused, precond.q_cov, out, gradient_component, hessian_component);
  }
}


inline void calc_phi_fused_last_step_preconditioned(const PHBackblaze_centered_grouped_halfnormal &model,
                                                    const arma::mat &fused,
                                                    const int nshards,
                                                    arma::vec &out,
                                                    arma::vec &gradient_component,
                                                    arma::vec &hessian_component,
                                                    const AbfPreconditioner &precond) {
  if (precond.mode == AbfPreconditioningMode::No) {
    model.calc_phi_fused_last_step(fused, nshards, out, gradient_component, hessian_component);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    model.calc_phi_fused_last_step_diagonal(fused, precond.q_diag, out, gradient_component, hessian_component);
  } else {
    model.calc_phi_fused_last_step_full(fused, precond.q_cov, out, gradient_component, hessian_component);
  }
}


struct LbbPoissonSegmentWeight {
  double log_weight;
  int kap;
  double bound_intensity;
};

inline LbbPoissonSegmentWeight lbb_poisson_segment_weight(
    const Logit &model,
    const Rcpp::Function &multi_bessel_layer_simulation,
    const Rcpp::Function &multi_layered_brownian_bridge,
    const arma::vec &x0,
    const arma::vec &x1,
    const double s,
    const double t,
    const int shard_1based,
    const int particle_1based,
    const int step_1based) {
  for (arma::uword r = 0; r < x0.n_elem; ++r) {
    if (!std::isfinite(x0[r])) {
      Rcpp::stop("lbb_poisson_segment_weight: nonfinite x0; particle=%d step=%d shard=%d s=%g t=%g",
                 particle_1based, step_1based, shard_1based, s, t);
    }
  }
  for (arma::uword r = 0; r < x1.n_elem; ++r) {
    if (!std::isfinite(x1[r])) {
      Rcpp::stop("lbb_poisson_segment_weight: nonfinite x1; particle=%d step=%d shard=%d s=%g t=%g",
                 particle_1based, step_1based, shard_1based, s, t);
    }
  }
  if (!std::isfinite(s) || !std::isfinite(t) || !(t > s)) {
    Rcpp::stop("lbb_poisson_segment_weight: expected finite times with t > s; particle=%d step=%d shard=%d s=%g t=%g",
               particle_1based, step_1based, shard_1based, s, t);
  }

  const int d = static_cast<int>(x0.n_elem);
  if (static_cast<int>(x1.n_elem) != d) {
    Rcpp::stop("lbb_poisson_segment_weight: endpoint dimensions differ; particle=%d step=%d shard=%d s=%g t=%g",
               particle_1based, step_1based, shard_1based, s, t);
  }

  const Rcpp::List bessel_layers = multi_bessel_layer_simulation(
      Rcpp::_["dim"] = d,
      Rcpp::_["x"] = x0,
      Rcpp::_["y"] = x1,
      Rcpp::_["s"] = s,
      Rcpp::_["t"] = t,
      Rcpp::_["mult"] = 0.05); //mult is the spacing between the layers of BB. The lerger, the faster they are to simulate, the higher the rejection.

  const Logit::LayerBounds bounds = model.global_layer_bounds(bessel_layers, shard_1based);
  const double LB = bounds.LB;
  const double UB = bounds.UB;
  if (!std::isfinite(LB) || !std::isfinite(UB) || !(UB > LB)) {
    Rcpp::stop("lbb_poisson_segment_weight: invalid bounds; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB);
  }

  const double bound_intensity = (UB - LB) * (t - s);
  if (!std::isfinite(bound_intensity) || bound_intensity < 0.0) {
    Rcpp::stop("lbb_poisson_segment_weight: invalid bound intensity; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
  }

  const double kap_draw = R::rpois(bound_intensity);
  if (!std::isfinite(kap_draw) || kap_draw < 0.0 ||
      kap_draw > static_cast<double>(std::numeric_limits<int>::max())) {
    Rcpp::stop("lbb_poisson_segment_weight: invalid Poisson draw; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
  }
  const int kap = static_cast<int>(kap_draw);

  if (kap == 0) {
    return LbbPoissonSegmentWeight{-LB * (t - s), kap, bound_intensity};
  }

  arma::vec times(kap);
  for (int j = 0; j < kap; ++j) {
    times[j] = R::runif(s, t);
  }

  const Rcpp::List layered_bb = multi_layered_brownian_bridge(
      Rcpp::_["dim"] = d,
      Rcpp::_["x"] = x0,
      Rcpp::_["y"] = x1,
      Rcpp::_["s"] = s,
      Rcpp::_["t"] = t,
      Rcpp::_["bessel_layers"] = bessel_layers,
      Rcpp::_["times"] = times);

  if (!layered_bb.containsElementNamed("simulated_path")) {
    Rcpp::stop("lbb_poisson_segment_weight: layeredBB result lacks simulated_path; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
  }
  SEXP path_sexp = layered_bb["simulated_path"];
  if (!Rf_isReal(path_sexp)) {
    Rcpp::stop("lbb_poisson_segment_weight: simulated_path is not numeric; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
  }
  const Rcpp::NumericMatrix simulated_path(path_sexp);
  if (simulated_path.nrow() != d + 1 || simulated_path.ncol() != kap) {
    Rcpp::stop("lbb_poisson_segment_weight: simulated_path has wrong dimensions; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
               particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
  }

  arma::mat path_matrix(kap, d);
  for (int j = 0; j < kap; ++j) {
    for (int r = 0; r < d; ++r) {
      path_matrix(j, r) = simulated_path(r, j);
    }
  }

  arma::vec phi_values(kap);
  model.calc_phi_for_all_rows(path_matrix, phi_values, shard_1based);

  double phi_min = std::numeric_limits<double>::infinity();
  double phi_max = -std::numeric_limits<double>::infinity();
  bool have_phi_range = false;
  double log_weight = -LB * (t - s) - static_cast<double>(kap) * std::log(UB - LB);

  for (int j = 0; j < kap; ++j) {
    const double phi_j = phi_values[j];
    if (std::isfinite(phi_j)) {
      if (!have_phi_range || phi_j < phi_min) {
        phi_min = phi_j;
      }
      if (!have_phi_range || phi_j > phi_max) {
        phi_max = phi_j;
      }
      have_phi_range = true;
    } else {
      if (have_phi_range) {
        Rcpp::stop("lbb_poisson_segment_weight: nonfinite phi; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g phi_min=%g phi_max=%g bound_intensity=%g",
                   particle_1based, step_1based, shard_1based, s, t, LB, UB, phi_min, phi_max, bound_intensity);
      }
      Rcpp::stop("lbb_poisson_segment_weight: nonfinite phi; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g bound_intensity=%g",
                 particle_1based, step_1based, shard_1based, s, t, LB, UB, bound_intensity);
    }

    if (phi_j < LB || phi_j > UB) {
      Rcpp::stop("lbb_poisson_segment_weight: phi outside bounds; particle=%d step=%d shard=%d s=%g t=%g LB=%g UB=%g phi_min=%g phi_max=%g bound_intensity=%g",
                 particle_1based, step_1based, shard_1based, s, t, LB, UB, phi_min, phi_max, bound_intensity);
    }

    log_weight += std::log(UB - phi_j);
  }

  return LbbPoissonSegmentWeight{log_weight, kap, bound_intensity};
}

struct BoundedRecommendations {
  double v_from_variance;
  double v_from_conditional_ess;
  double delta_ess_raw;
  double delta_ess_bounded;
  double delta_variance_raw;
  double delta_variance_bounded;
};

struct AsymptoticStepSizeRecommendation {
  double expected_nu;
  double k3;
  double k4;
  double delta_term1;
  double delta_term2;
  int selected_term;
  double delta_raw;
  double delta_bounded;
};

inline void validate_step_size_inputs(const double target_one_step_ess,
                                      const double step_size_min_factor,
                                      const double step_size_max_factor,
                                      const double step_size_total_min) {
  if (!std::isfinite(target_one_step_ess) ||
      target_one_step_ess <= 0.0 ||
      target_one_step_ess >= 1.0) {
    Rcpp::stop("target_one_step_ess must lie in (0, 1).");
  }
  if (!std::isfinite(step_size_min_factor) ||
      !std::isfinite(step_size_max_factor) ||
      step_size_min_factor <= 0.0 ||
      step_size_min_factor > 1.0 ||
      step_size_max_factor < 1.0) {
    Rcpp::stop("step size factors must satisfy 0 < step_size_min_factor <= 1 <= step_size_max_factor.");
  }
  if (!std::isfinite(step_size_total_min) || step_size_total_min < 0.0) {
    Rcpp::stop("step_size_total_min must be a finite non-negative number.");
  }
}

inline std::vector<double> validate_time_grid(const SEXP time_grid,
                                              const double end_time) {
  if (TYPEOF(time_grid) != INTSXP && TYPEOF(time_grid) != REALSXP) {
    Rcpp::stop("time_grid must be an unclassed integer or double vector.");
  }
  if (Rf_getAttrib(time_grid, R_DimSymbol) != R_NilValue) {
    Rcpp::stop("time_grid must be a dimensionless integer or double vector.");
  }
  if (Rf_getAttrib(time_grid, R_ClassSymbol) != R_NilValue) {
    Rcpp::stop("time_grid must be an unclassed integer or double vector.");
  }

  const R_xlen_t n = XLENGTH(time_grid);
  if (n < 2) {
    Rcpp::stop("time_grid must contain at least two entries.");
  }

  std::vector<double> out(static_cast<std::size_t>(n));
  for (R_xlen_t i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = TYPEOF(time_grid) == INTSXP
      ? static_cast<double>(INTEGER(time_grid)[i])
      : REAL(time_grid)[i];
    if (!std::isfinite(out[static_cast<std::size_t>(i)])) {
      Rcpp::stop("time_grid must contain only finite values.");
    }
  }

  if (out.front() != 0.0) {
    Rcpp::stop("time_grid must start at exactly 0.0.");
  }
  if (out.back() != end_time) {
    Rcpp::stop("time_grid must end at exactly end_time.");
  }
  for (std::size_t i = 1; i < out.size(); ++i) {
    if (!(out[i] > out[i - 1])) {
      Rcpp::stop("time_grid must be strictly increasing.");
    }
  }
  return out;
}

inline double endpoint_tolerance(const double end_time,
                                 const double previous_t,
                                 const double remaining,
                                 const double delta_next,
                                 const int completed_ordinary_steps) {
  double scale = std::abs(end_time);
  if (std::abs(previous_t) > scale) scale = std::abs(previous_t);
  if (std::abs(remaining) > scale) scale = std::abs(remaining);
  if (std::isfinite(delta_next) && std::abs(delta_next) > scale) scale = std::abs(delta_next);
  const int step_factor = completed_ordinary_steps + 1 > 1 ? completed_ordinary_steps + 1 : 1;
  return static_cast<double>(step_factor) * abf_round_tolerance(scale);
}

inline double cap_and_snap_to_remaining(const double proposed,
                                        const double remaining,
                                        const double endpoint_tol) {
  if (proposed >= remaining) {
    return remaining;
  }
  if (remaining > proposed && remaining - proposed <= endpoint_tol) {
    return remaining;
  }
  return proposed;
}

inline Rcpp::LogicalVector logical_vector_from_ints(const std::vector<int> &x) {
  Rcpp::LogicalVector out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] != 0;
  }
  return out;
}

inline BoundedRecommendations calculate_bounded_recommendations(
    const IncrementalWeightStats &stats,
    const double delta_used,
    const double target_one_step_ess,
    const double step_size_min_factor,
    const double step_size_max_factor,
    const double step_size_total_min,
    const double remaining,
    const double endpoint_tol) {
  BoundedRecommendations out;
  const double log_target = std::log(target_one_step_ess);
  const double log_zeta = std::log1p(stats.conditional_ess_ratio - 1.0);

  out.v_from_conditional_ess = -log_zeta / (delta_used * delta_used);
  if (1.0 - stats.conditional_ess_ratio <= abf_round_tolerance(1.0)) {
    out.delta_ess_raw = std::numeric_limits<double>::infinity();
    double delta_clip = step_size_max_factor * delta_used;
    if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
    out.delta_ess_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
  } else {
    out.delta_ess_raw = delta_used * std::sqrt(log_target / log_zeta);
    double delta_clip = out.delta_ess_raw;
    const double lower = step_size_min_factor * delta_used;
    const double upper = step_size_max_factor * delta_used;
    if (delta_clip < lower) delta_clip = lower;
    if (delta_clip > upper) delta_clip = upper;
    if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
    out.delta_ess_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
  }

  out.v_from_variance = stats.logweight_variance / (delta_used * delta_used);
  if (stats.logweight_variance <= abf_round_tolerance(stats.variance_scale)) {
    out.delta_variance_raw = std::numeric_limits<double>::infinity();
    double delta_clip = step_size_max_factor * delta_used;
    if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
    out.delta_variance_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
  } else {
    out.delta_variance_raw = std::sqrt(-log_target / out.v_from_variance);
    double delta_clip = out.delta_variance_raw;
    const double lower = step_size_min_factor * delta_used;
    const double upper = step_size_max_factor * delta_used;
    if (delta_clip < lower) delta_clip = lower;
    if (delta_clip > upper) delta_clip = upper;
    if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
    out.delta_variance_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
  }

  return out;
}

inline arma::mat calculate_direct_subposterior_means(const arma::cube &particles) {
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  arma::mat means(nshards, ndim, arma::fill::zeros);
  for (int c = 0; c < nshards; ++c) {
    for (int d = 0; d < ndim; ++d) {
      means(c, d) = arma::mean(particles.slice(d).col(c));
    }
  }
  return means;
}

inline arma::mat make_subposterior_means(SEXP subposterior_means,
                                         const arma::cube &particles) {
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  if (Rf_isNull(subposterior_means)) {
    return calculate_direct_subposterior_means(particles);
  }
  arma::mat means = Rcpp::as<arma::mat>(subposterior_means);
  check_abf_mat_shape(means, nshards, ndim, "subposterior_means");
  return means;
}

// The variance s^2 = C b / m of each coordinate of the subposteriors, which the
// asymptotic rule of Dai et al. (2023) assumes to be N(a_c, s^2 I_d). By
// default it is the average over shards and coordinates of the variances of the
// particles. Preconditioning whitens the subposteriors, so that s^2 = 1.
inline double make_subposterior_variance(SEXP subposterior_variance,
                                         const arma::cube &particles,
                                         const AbfPreconditioner &precond) {
  if (precond.mode != AbfPreconditioningMode::No) {
    return 1.0;
  }
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  if (Rf_isNull(subposterior_variance)) {
    double total = 0.0;
    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        total += arma::var(particles.slice(d).col(c));
      }
    }
    return total / static_cast<double>(nshards * ndim);
  }
  const double variance = Rcpp::as<double>(subposterior_variance);
  if (!std::isfinite(variance) || variance <= 0.0) {
    Rcpp::stop("subposterior_variance must be a positive finite scalar.");
  }
  return variance;
}

inline AsymptoticStepSizeRecommendation calculate_asymptotic_step_size_recommendation(
    const arma::cube &particles,
    const arma::vec &weights,
    const arma::mat &subposterior_means,
    const double subposterior_variance,
    const AbfPreconditioner &precond,
    const double target_one_step_ess,
    const double bridge_time,
    const double delta_used,
    const double step_size_min_factor,
    const double step_size_max_factor,
    const double step_size_total_min,
    const double remaining,
    const double endpoint_tol) {
  check_normalized_weights(weights);

  const int npart = static_cast<int>(particles.n_rows);
  const int nshards = static_cast<int>(particles.n_cols);
  const int ndim = static_cast<int>(particles.n_slices);
  arma::vec nu(npart, arma::fill::zeros);
  arma::rowvec diff(ndim, arma::fill::zeros);

  for (int i = 0; i < npart; ++i) {
    double nu_i = 0.0;
    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        diff[d] = particles(i, c, d) - subposterior_means(c, d);
      }
      if (precond.mode == AbfPreconditioningMode::No) {
        nu_i += arma::dot(diff, diff);
      } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
        double shard_sum = 0.0;
        for (int d = 0; d < ndim; ++d) {
          shard_sum += precond.q_inv_diag(c, d) * diff[d] * diff[d];
        }
        nu_i += shard_sum;
      } else {
        nu_i += arma::as_scalar(diff * precond.q_inv[c] * diff.t());
      }
    }
    nu[i] = nu_i / static_cast<double>(nshards);
  }

  AsymptoticStepSizeRecommendation out;
  out.expected_nu = arma::dot(weights, nu);
  const double nu_scale = nu.max();
  const double A = -std::log(target_one_step_ess);
  // Equation (21) of Dai et al. (2023) with s^2 = C b / m is s^2 times the
  // same equation with s^2 = 1 and E nu replaced by E nu / s^2.
  const double s2 = subposterior_variance;
  const double scaled_expected_nu = out.expected_nu / s2;

  if (out.expected_nu <= abf_round_tolerance(nu_scale)) {
    out.k3 = 0.0;
    out.k4 = A;
    out.delta_term1 = s2 * std::sqrt(A / (2.0 * static_cast<double>(nshards) * static_cast<double>(ndim)));
    out.delta_term2 = out.delta_term1;
    out.selected_term = 2;
    out.delta_raw = out.delta_term2;
    double delta_clip = out.delta_raw;
    const double lower = step_size_min_factor * delta_used;
    const double upper = step_size_max_factor * delta_used;
    if (delta_clip < lower) delta_clip = lower;
    if (delta_clip > upper) delta_clip = upper;
    if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
    out.delta_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
    return out;
  }

  const double B = 2.0 * A +
    static_cast<double>(nshards) * scaled_expected_nu * scaled_expected_nu /
    (2.0 * static_cast<double>(ndim));
  double discriminant = (B - 2.0 * A) * (B + 2.0 * A);
  double discriminant_scale = B * B;
  const double four_a_sq = 4.0 * A * A;
  if (four_a_sq > discriminant_scale) discriminant_scale = four_a_sq;
  if (discriminant < 0.0 &&
      discriminant >= -abf_round_tolerance(discriminant_scale)) {
    discriminant = 0.0;
  }
  if (discriminant < 0.0) {
    Rcpp::stop("negative asymptotic step-size discriminant at bridge time %.17g: A=%.17g, B=%.17g, D=%.17g.",
               bridge_time, A, B, discriminant);
  }

  const double sqrt_discriminant = std::sqrt(discriminant);
  const double root_denominator = B + sqrt_discriminant;
  out.k4 = 2.0 * A * A / root_denominator;
  out.k3 = A * ((B - 2.0 * A) + sqrt_discriminant) / root_denominator;
  out.delta_term1 = s2 * out.k3 / (static_cast<double>(nshards) * scaled_expected_nu);
  out.delta_term2 = s2 * std::sqrt(out.k4 / (2.0 * static_cast<double>(nshards) * static_cast<double>(ndim)));
  double term_scale = std::abs(out.delta_term1);
  if (std::abs(out.delta_term2) > term_scale) term_scale = std::abs(out.delta_term2);
  if (out.delta_term1 < out.delta_term2 &&
      out.delta_term2 - out.delta_term1 > abf_round_tolerance(term_scale)) {
    out.selected_term = 1;
    out.delta_raw = out.delta_term1;
  } else {
    out.selected_term = 2;
    out.delta_raw = out.delta_term2;
  }
  double delta_clip = out.delta_raw;
  const double lower = step_size_min_factor * delta_used;
  const double upper = step_size_max_factor * delta_used;
  if (delta_clip < lower) delta_clip = lower;
  if (delta_clip > upper) delta_clip = upper;
  if (delta_clip < step_size_total_min) delta_clip = step_size_total_min;
  out.delta_bounded = cap_and_snap_to_remaining(delta_clip, remaining, endpoint_tol);
  return out;
}

// -------------------------------------------------------------------------
// Generic SMC engine driven by a compiled model object. The model class must
// provide calc_phi_cube_layer_sums{,_diagonal,_full} and
// calc_phi_fused_last_step{,_diagonal,_full}, which also return the gradient
// and Hessian parts of phi (see model_classes.hpp).
// -------------------------------------------------------------------------

// The incremental log weight over a step of size delta is
// -(phi(x_new) + phi(x_old)) * delta / 2, so its variance across particles is
// delta^2 * var(phi) + O(delta^3) and the step meeting the one-step ESS target
// satisfies -log(target) = delta^2 * var(phi). Half of that value is used, so
// that the first step falls below the target and the in-loop rule only has to
// coarsen the step, which it can do safely, instead of recovering from weights
// that have already degenerated.
inline double first_step_size_from_phi(const arma::vec &phi_row_sum,
                                       const double target_one_step_ess) {
  const double phi_var = arma::var(phi_row_sum);
  if (!std::isfinite(phi_var) || phi_var <= 0.0) {
    Rcpp::stop("cannot choose the first step size: phi has no spread across the initial particles.");
  }
  return 0.5 * std::sqrt(-std::log(target_one_step_ess) / phi_var);
}

template <typename Model>
Rcpp::List run_bayesian_fusion_smc(const Model &model,
                             const std::string &model_name,
                             const double end_time,
                             const int nchunks,
                             arma::cube &init_particles,
                             SEXP &init_weight,
                             const bool resamp,
                             const double resamp_thr,
                             const bool only_one_step,
                             const bool align_phi_on_resample,
                             const AbfPreconditioner &precond,
                             const double target_one_step_ess,
                             const double step_size_min_factor,
                             const double step_size_max_factor,
                             const double step_size_total_min,
                             const std::string &step_size_criterion,
                             const arma::mat &subposterior_means,
                             const double subposterior_variance,
                             const int effective_n_threads,
                             const std::vector<double> &time_grid,
                             const bool use_fixed_grid,
                             const bool auto_first_step) {
  Rcpp::RNGScope scope;

  const int npart = static_cast<int>(init_particles.n_rows);
  const int nshards = static_cast<int>(init_particles.n_cols);
  const int ndim = static_cast<int>(init_particles.n_slices);

  int resampcount = 0;
  const std::size_t reserve_n = use_fixed_grid ? time_grid.size() - 1 :
                                               static_cast<std::size_t>(nchunks);
  std::vector<double> time_mesh;
  std::vector<double> essvec;
  std::vector<double> phi_gradient_component_estimate;
  std::vector<double> phi_hessian_component_estimate;
  std::vector<double> phi_gradient_component_unweighted_estimate_q01;
  std::vector<double> phi_gradient_component_unweighted_estimate_q99;
  std::vector<double> phi_hessian_component_unweighted_estimate_q01;
  std::vector<double> phi_hessian_component_unweighted_estimate_q99;
  std::vector<double> phi_unweighted_estimate_q01;
  std::vector<double> phi_unweighted_estimate_q99;
  std::vector<double> phi_weighted_estimate_q01;
  std::vector<double> phi_weighted_estimate_q99;
  std::vector<double> phi_unweighted_estimate;
  std::vector<double> phi_weighted_estimate;
  std::vector<double> time_before_step;
  std::vector<double> time_after_step;
  std::vector<double> delta_used_vec;
  std::vector<int> resampled_before_step_vec;
  std::vector<int> resampled_before_next_step_vec;
  std::vector<double> conditional_ess_ratio;
  std::vector<double> conditional_ess;
  std::vector<double> logweight_variance;
  std::vector<double> log_incremental_weight_sd;
  std::vector<double> log_incremental_weight_q01;
  std::vector<double> log_incremental_weight_q50;
  std::vector<double> log_incremental_weight_q99;
  std::vector<double> log_incremental_weight_q99_q01;
  std::vector<double> v_from_variance;
  std::vector<double> v_from_conditional_ess;
  std::vector<double> delta_ess_raw;
  std::vector<double> delta_ess_bounded;
  std::vector<double> delta_variance_raw;
  std::vector<double> delta_variance_bounded;
  std::vector<double> asymptotic_expected_nu;
  std::vector<double> asymptotic_k3;
  std::vector<double> asymptotic_k4;
  std::vector<double> asymptotic_delta_term1;
  std::vector<double> asymptotic_delta_term2;
  std::vector<int> asymptotic_selected_term;
  std::vector<double> delta_asymptotic_raw;
  std::vector<double> delta_asymptotic_bounded;
  time_mesh.reserve(reserve_n + 1);
  essvec.reserve(reserve_n + 1);
  phi_gradient_component_estimate.reserve(reserve_n);
  phi_hessian_component_estimate.reserve(reserve_n);
  phi_gradient_component_unweighted_estimate_q01.reserve(reserve_n);
  phi_gradient_component_unweighted_estimate_q99.reserve(reserve_n);
  phi_hessian_component_unweighted_estimate_q01.reserve(reserve_n);
  phi_hessian_component_unweighted_estimate_q99.reserve(reserve_n);
  phi_unweighted_estimate_q01.reserve(reserve_n);
  phi_unweighted_estimate_q99.reserve(reserve_n);
  phi_weighted_estimate_q01.reserve(reserve_n);
  phi_weighted_estimate_q99.reserve(reserve_n);
  phi_unweighted_estimate.reserve(reserve_n);
  phi_weighted_estimate.reserve(reserve_n);
  time_before_step.reserve(reserve_n);
  time_after_step.reserve(reserve_n);
  delta_used_vec.reserve(reserve_n);
  resampled_before_step_vec.reserve(reserve_n);
  resampled_before_next_step_vec.reserve(reserve_n);
  conditional_ess_ratio.reserve(reserve_n);
  conditional_ess.reserve(reserve_n);
  logweight_variance.reserve(reserve_n);
  log_incremental_weight_sd.reserve(reserve_n);
  log_incremental_weight_q01.reserve(reserve_n);
  log_incremental_weight_q50.reserve(reserve_n);
  log_incremental_weight_q99.reserve(reserve_n);
  log_incremental_weight_q99_q01.reserve(reserve_n);
  v_from_variance.reserve(reserve_n);
  v_from_conditional_ess.reserve(reserve_n);
  delta_ess_raw.reserve(reserve_n);
  delta_ess_bounded.reserve(reserve_n);
  delta_variance_raw.reserve(reserve_n);
  delta_variance_bounded.reserve(reserve_n);
  asymptotic_expected_nu.reserve(reserve_n);
  asymptotic_k3.reserve(reserve_n);
  asymptotic_k4.reserve(reserve_n);
  asymptotic_delta_term1.reserve(reserve_n);
  asymptotic_delta_term2.reserve(reserve_n);
  asymptotic_selected_term.reserve(reserve_n);
  delta_asymptotic_raw.reserve(reserve_n);
  delta_asymptotic_bounded.reserve(reserve_n);
  arma::vec nlogw(npart, arma::fill::zeros);
  arma::vec normw(npart, arma::fill::zeros);
  arma::vec phi_row_sum(npart, arma::fill::zeros);
  arma::vec phi_row_sum_old(npart, arma::fill::zeros);
  arma::vec phi_gradient_component(npart, arma::fill::zeros);
  arma::vec phi_hessian_component(npart, arma::fill::zeros);
  arma::cube x_old = init_particles;
  arma::cube x_new(npart, nshards, ndim, arma::fill::zeros);
  arma::mat x_bar(npart, ndim, arma::fill::zeros);
  arma::mat out(npart, ndim, arma::fill::zeros);
  arma::mat bridge_psi(npart, nshards, arma::fill::zeros);
  arma::vec bridge_chi(npart, arma::fill::zeros);
  arma::vec bridge_shift(npart, arma::fill::zeros);


  if (precond.mode == AbfPreconditioningMode::No) {
    compute_shard_means(x_old, x_bar);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    compute_weighted_centres_diagonal(x_old, x_bar, precond);
  } else {
    compute_weighted_centres_full(x_old, x_bar, precond);
  }

  set_initial_weights(x_old, x_bar, end_time, init_weight, nlogw, normw, &precond);
  
  calc_phi_cube_layer_sums_preconditioned(model, x_old, phi_row_sum_old,
                                          phi_gradient_component, phi_hessian_component,
                                          precond); // fill vec phi_row_sum_old with model specific phi summed across C and d

  const double initial_step_size = use_fixed_grid
    ? time_grid[1] - time_grid[0]
    : (auto_first_step
         ? first_step_size_from_phi(phi_row_sum_old, target_one_step_ess)
         : end_time / static_cast<double>(nchunks));
  double delta_next = 0.0;
  if (!use_fixed_grid) {
    delta_next = initial_step_size;
    if (delta_next < step_size_total_min) delta_next = step_size_total_min;
  }
  double previous_t = 0.0;

  double ess = essw(normw);
  time_mesh.push_back(use_fixed_grid ? time_grid[0] : 0.0);
  essvec.push_back(ess);

  auto resample_current_state = [&]() {
    ++resampcount;
    const arma::uvec idx = systematic_residual_resampling(normw);

    update_part_after_resamp(x_old, idx, x_new);
    x_old.swap(x_new);

    if (align_phi_on_resample) { // update phi after resampling the positions
      update_phi_reorder_rows(phi_row_sum_old, idx, phi_row_sum);
      phi_row_sum_old.swap(phi_row_sum);
    }

    normw.fill(1.0 / static_cast<double>(npart));
    nlogw = -arma::log(normw);
  };

  bool resampled_before_step = false;
  if (resamp && ess < static_cast<double>(npart) * resamp_thr) {
    resample_current_state();
    resampled_before_step = true;
  }

  int completed_ordinary_steps = 0;
  while (true) {
    Rcpp::checkUserInterrupt();

    const int ordinary_step = completed_ordinary_steps + 1;
    double delta_used;
    double current_t;
    if (use_fixed_grid) {
      if (completed_ordinary_steps >= static_cast<int>(time_grid.size()) - 2) {
        break;
      }
      previous_t = time_grid[static_cast<std::size_t>(ordinary_step - 1)];
      current_t = time_grid[static_cast<std::size_t>(ordinary_step)];
      delta_used = current_t - previous_t;
    } else {
      const double remaining_before_step = end_time - previous_t;
      const double terminal_tol = endpoint_tolerance(
          end_time, previous_t, remaining_before_step, delta_next,
          completed_ordinary_steps);
      if (delta_next >= remaining_before_step ||
          (remaining_before_step > delta_next &&
           remaining_before_step - delta_next <= terminal_tol)) {
        break;
      }

      delta_used = delta_next;
      current_t = previous_t + delta_used;
    }

    const double remaining_before_step = end_time - previous_t;
    if (!std::isfinite(delta_used) || delta_used <= 0.0) {
      Rcpp::stop("invalid proposed interval at ordinary step %d: current time %.17g, remaining time %.17g, proposed interval %.17g.",
                 ordinary_step, previous_t, remaining_before_step, delta_used);
    }
    if (!(current_t > previous_t)) {
      Rcpp::stop("time did not advance at ordinary step %d: current time %.17g, remaining time %.17g, proposed interval %.17g.",
                 ordinary_step, previous_t, remaining_before_step, delta_used);
    }
    if (current_t > end_time) {
      Rcpp::stop("ordinary step overshot the endpoint at ordinary step %d: current time %.17g, remaining time %.17g, proposed interval %.17g.",
                 ordinary_step, previous_t, remaining_before_step, delta_used);
    }
    const double recorded_time_before_step = use_fixed_grid
      ? previous_t
      : current_t - delta_used;

    const double alpha = (end_time - current_t) / (end_time - previous_t);
    const double beta = delta_used / (end_time - previous_t);

    if (precond.mode == AbfPreconditioningMode::No) {
      propagate_bridge(x_old, x_new, x_bar, bridge_psi, bridge_chi, bridge_shift, alpha, beta, delta_used);
    } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
      propagate_bridge_diagonal(x_old, x_new, x_bar, precond, alpha, beta, delta_used);
    } else {
      propagate_bridge_full(x_old, x_new, x_bar, precond, alpha, beta, delta_used);
    }
    calc_phi_cube_layer_sums_preconditioned(model, x_new, phi_row_sum,
                                            phi_gradient_component, phi_hessian_component,
                                            precond); // fill vec phi_row_sum with model specific phi at new position x_new.

    const arma::vec negative_log_increment = 0.5 * (phi_row_sum + phi_row_sum_old) * delta_used;
    const arma::vec log_incremental_weights = -negative_log_increment;
    const IncrementalWeightStats weight_stats = calculate_incremental_weight_stats(
        normw, log_incremental_weights, ordinary_step, current_t);
    const double remaining_after_step = end_time - current_t;
    const double endpoint_tol = endpoint_tolerance(
        end_time, current_t, remaining_after_step, delta_used,
        ordinary_step);
    const BoundedRecommendations rec = calculate_bounded_recommendations(
        weight_stats, delta_used, target_one_step_ess, step_size_min_factor,
        step_size_max_factor, step_size_total_min, remaining_after_step, endpoint_tol);

    const arma::vec sorted_gradient_component = arma::sort(phi_gradient_component);
    const arma::vec sorted_hessian_component = arma::sort(phi_hessian_component);
    phi_gradient_component_unweighted_estimate_q01.push_back(
        linear_quantile(sorted_gradient_component, 0.01));
    phi_gradient_component_unweighted_estimate_q99.push_back(
        linear_quantile(sorted_gradient_component, 0.99));
    phi_hessian_component_unweighted_estimate_q01.push_back(
        linear_quantile(sorted_hessian_component, 0.01));
    phi_hessian_component_unweighted_estimate_q99.push_back(
        linear_quantile(sorted_hessian_component, 0.99));
    const arma::uvec phi_order = arma::sort_index(phi_row_sum);
    const arma::vec sorted_phi = phi_row_sum.elem(phi_order);
    phi_unweighted_estimate_q01.push_back(linear_quantile(sorted_phi, 0.01));
    phi_unweighted_estimate_q99.push_back(linear_quantile(sorted_phi, 0.99));
    phi_unweighted_estimate.push_back(arma::mean(phi_row_sum));

    nlogw += negative_log_increment;
    normw = calc_weights(-nlogw);
    const arma::vec sorted_phi_weights = normw.elem(phi_order);
    phi_weighted_estimate_q01.push_back(
        weighted_empirical_quantile(sorted_phi, sorted_phi_weights, 0.01));
    phi_weighted_estimate_q99.push_back(
        weighted_empirical_quantile(sorted_phi, sorted_phi_weights, 0.99));
    phi_weighted_estimate.push_back(arma::dot(normw, phi_row_sum));
    const double gradient_estimate = arma::dot(normw, phi_gradient_component);
    const double hessian_estimate = arma::dot(normw, phi_hessian_component);
    if (!std::isfinite(gradient_estimate) || !std::isfinite(hessian_estimate)) {
      Rcpp::stop("PHI component estimate is not finite.");
    }
    phi_gradient_component_estimate.push_back(gradient_estimate);
    phi_hessian_component_estimate.push_back(hessian_estimate);

    phi_row_sum_old.swap(phi_row_sum); // swap instead of copying over, faster
    x_old.swap(x_new); //non-old becomes buffer. State propagates from old and populates new.

    previous_t = current_t;

    ess = essw(normw);
    time_mesh.push_back(current_t);
    essvec.push_back(ess);

    if (only_one_step) {
      arma::mat out_one(npart, ndim, arma::fill::zeros);
      if (precond.mode == AbfPreconditioningMode::No) {
        compute_shard_means(x_old, x_bar);
      } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
        compute_weighted_centres_diagonal(x_old, x_bar, precond);
      } else {
        compute_weighted_centres_full(x_old, x_bar, precond);
      }
      out_one = x_bar;
      Rcpp::List one_step_result;
      one_step_result.push_back(out_one, "out");
      one_step_result.push_back(normw, "normw");
      one_step_result.push_back(end_time, "end_time");
      one_step_result.push_back(time_mesh, "time_mesh");
      one_step_result.push_back(essvec, "essvec");
      // PHI component estimates use time_mesh without its initial entry.
      one_step_result.push_back(phi_gradient_component_estimate, "phi_gradient_component_estimate");
      one_step_result.push_back(phi_hessian_component_estimate, "phi_hessian_component_estimate");
      one_step_result.push_back(phi_gradient_component_unweighted_estimate_q01, "phi_gradient_component_unweighted_estimate_q01");
      one_step_result.push_back(phi_gradient_component_unweighted_estimate_q99, "phi_gradient_component_unweighted_estimate_q99");
      one_step_result.push_back(phi_hessian_component_unweighted_estimate_q01, "phi_hessian_component_unweighted_estimate_q01");
      one_step_result.push_back(phi_hessian_component_unweighted_estimate_q99, "phi_hessian_component_unweighted_estimate_q99");
      one_step_result.push_back(phi_unweighted_estimate_q01, "phi_unweighted_estimate_q01");
      one_step_result.push_back(phi_unweighted_estimate_q99, "phi_unweighted_estimate_q99");
      one_step_result.push_back(phi_weighted_estimate_q01, "phi_weighted_estimate_q01");
      one_step_result.push_back(phi_weighted_estimate_q99, "phi_weighted_estimate_q99");
      one_step_result.push_back(phi_unweighted_estimate, "phi_unweighted_estimate");
      one_step_result.push_back(phi_weighted_estimate, "phi_weighted_estimate");
      one_step_result.push_back(std::vector<double>{recorded_time_before_step}, "time_before_step");
      one_step_result.push_back(std::vector<double>{current_t}, "time_after_step");
      one_step_result.push_back(std::vector<double>{delta_used}, "delta_used");
      one_step_result.push_back(Rcpp::LogicalVector::create(resampled_before_step), "resampled_before_step");
      one_step_result.push_back(std::vector<double>{weight_stats.conditional_ess_ratio}, "conditional_ess_ratio");
      one_step_result.push_back(std::vector<double>{weight_stats.conditional_ess}, "conditional_ess");
      one_step_result.push_back(std::vector<double>{weight_stats.logweight_variance}, "logweight_variance");
      one_step_result.push_back(std::vector<double>{weight_stats.log_incremental_weight_sd}, "log_incremental_weight_sd");
      one_step_result.push_back(std::vector<double>{weight_stats.log_incremental_weight_q01}, "log_incremental_weight_q01");
      one_step_result.push_back(std::vector<double>{weight_stats.log_incremental_weight_q50}, "log_incremental_weight_q50");
      one_step_result.push_back(std::vector<double>{weight_stats.log_incremental_weight_q99}, "log_incremental_weight_q99");
      one_step_result.push_back(std::vector<double>{weight_stats.log_incremental_weight_q99_q01}, "log_incremental_weight_q99_q01");
      one_step_result.push_back(auto_first_step ? NA_INTEGER : nchunks, "initial_nchunks");
      one_step_result.push_back(initial_step_size, "initial_step_size");
      one_step_result.push_back(1L, "n_ordinary_steps");
      one_step_result.push_back(1L, "n_total_steps");
      one_step_result.push_back(target_one_step_ess, "target_one_step_ess");
      one_step_result.push_back(step_size_min_factor, "step_size_min_factor");
      one_step_result.push_back(step_size_max_factor, "step_size_max_factor");
      one_step_result.push_back(step_size_total_min, "step_size_total_min");
      one_step_result.push_back(resampcount, "resampcount");
      one_step_result.push_back(align_phi_on_resample, "align_phi_on_resample");
      one_step_result.push_back(model_name, "model");
      one_step_result.push_back(effective_n_threads, "n_threads");
      one_step_result.push_back(std::string("online"), "step_size_mode");
      return one_step_result;
    }

    bool resampled_before_next_step = false;
    if (resamp && ess < static_cast<double>(npart) * resamp_thr) {
      resample_current_state();
      resampled_before_next_step = true;
    }

    const AsymptoticStepSizeRecommendation asymptotic_step = calculate_asymptotic_step_size_recommendation(
        x_old, normw, subposterior_means, subposterior_variance, precond, target_one_step_ess,
        previous_t, delta_used, step_size_min_factor, step_size_max_factor,
        step_size_total_min, remaining_after_step, endpoint_tol);

    time_before_step.push_back(recorded_time_before_step);
    time_after_step.push_back(current_t);
    delta_used_vec.push_back(delta_used);
    resampled_before_step_vec.push_back(resampled_before_step ? 1 : 0);
    resampled_before_next_step_vec.push_back(resampled_before_next_step ? 1 : 0);
    conditional_ess_ratio.push_back(weight_stats.conditional_ess_ratio);
    conditional_ess.push_back(weight_stats.conditional_ess);
    logweight_variance.push_back(weight_stats.logweight_variance);
    log_incremental_weight_sd.push_back(weight_stats.log_incremental_weight_sd);
    log_incremental_weight_q01.push_back(weight_stats.log_incremental_weight_q01);
    log_incremental_weight_q50.push_back(weight_stats.log_incremental_weight_q50);
    log_incremental_weight_q99.push_back(weight_stats.log_incremental_weight_q99);
    log_incremental_weight_q99_q01.push_back(weight_stats.log_incremental_weight_q99_q01);
    v_from_variance.push_back(rec.v_from_variance);
    v_from_conditional_ess.push_back(rec.v_from_conditional_ess);
    delta_ess_raw.push_back(rec.delta_ess_raw);
    delta_ess_bounded.push_back(rec.delta_ess_bounded);
    delta_variance_raw.push_back(rec.delta_variance_raw);
    delta_variance_bounded.push_back(rec.delta_variance_bounded);
    asymptotic_expected_nu.push_back(asymptotic_step.expected_nu);
    asymptotic_k3.push_back(asymptotic_step.k3);
    asymptotic_k4.push_back(asymptotic_step.k4);
    asymptotic_delta_term1.push_back(asymptotic_step.delta_term1);
    asymptotic_delta_term2.push_back(asymptotic_step.delta_term2);
    asymptotic_selected_term.push_back(asymptotic_step.selected_term);
    delta_asymptotic_raw.push_back(asymptotic_step.delta_raw);
    delta_asymptotic_bounded.push_back(asymptotic_step.delta_bounded);

    if (!use_fixed_grid) {
      if (step_size_criterion == "ess_ratio") {
        delta_next = rec.delta_ess_bounded;
      } else {
        // The exported dispatcher validates step_size_criterion before this loop.
        delta_next = asymptotic_step.delta_bounded;
      }
    }
    resampled_before_step = resampled_before_next_step;
    completed_ordinary_steps = ordinary_step;
  }

  const double delta_terminal = use_fixed_grid
    ? time_grid.back() - time_grid[time_grid.size() - 2]
    : end_time - previous_t;
  if (!std::isfinite(delta_terminal) || delta_terminal <= 0.0) {
    Rcpp::stop("invalid terminal interval after %d ordinary steps: current time %.17g, remaining time %.17g, proposed interval %.17g.",
               completed_ordinary_steps, previous_t, delta_terminal, delta_next);
  }
  if (precond.mode == AbfPreconditioningMode::No) {
    sample_fused_endpoint(x_old, x_bar, out, delta_terminal);
  } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
    sample_fused_endpoint_diagonal(x_old, x_bar, out, precond, delta_terminal);
  } else {
    sample_fused_endpoint_full(x_old, x_bar, out, precond, delta_terminal);
  }
  calc_phi_fused_last_step_preconditioned(model, out, nshards, phi_row_sum,
                                          phi_gradient_component, phi_hessian_component,
                                          precond); // update phi_row_sum

  const arma::vec sorted_gradient_component = arma::sort(phi_gradient_component);
  const arma::vec sorted_hessian_component = arma::sort(phi_hessian_component);
  phi_gradient_component_unweighted_estimate_q01.push_back(
      linear_quantile(sorted_gradient_component, 0.01));
  phi_gradient_component_unweighted_estimate_q99.push_back(
      linear_quantile(sorted_gradient_component, 0.99));
  phi_hessian_component_unweighted_estimate_q01.push_back(
      linear_quantile(sorted_hessian_component, 0.01));
  phi_hessian_component_unweighted_estimate_q99.push_back(
      linear_quantile(sorted_hessian_component, 0.99));
  const arma::uvec phi_order = arma::sort_index(phi_row_sum);
  const arma::vec sorted_phi = phi_row_sum.elem(phi_order);
  phi_unweighted_estimate_q01.push_back(linear_quantile(sorted_phi, 0.01));
  phi_unweighted_estimate_q99.push_back(linear_quantile(sorted_phi, 0.99));
  phi_unweighted_estimate.push_back(arma::mean(phi_row_sum));

  const arma::vec negative_log_increment = 0.5 * (phi_row_sum + phi_row_sum_old) * delta_terminal;
  nlogw += negative_log_increment;
  
  normw = calc_weights(-nlogw);
  const arma::vec sorted_phi_weights = normw.elem(phi_order);
  phi_weighted_estimate_q01.push_back(
      weighted_empirical_quantile(sorted_phi, sorted_phi_weights, 0.01));
  phi_weighted_estimate_q99.push_back(
      weighted_empirical_quantile(sorted_phi, sorted_phi_weights, 0.99));
  phi_weighted_estimate.push_back(arma::dot(normw, phi_row_sum));
  const double gradient_estimate = arma::dot(normw, phi_gradient_component);
  const double hessian_estimate = arma::dot(normw, phi_hessian_component);
  if (!std::isfinite(gradient_estimate) || !std::isfinite(hessian_estimate)) {
    Rcpp::stop("PHI component estimate is not finite.");
  }
  phi_gradient_component_estimate.push_back(gradient_estimate);
  phi_hessian_component_estimate.push_back(hessian_estimate);
  time_mesh.push_back(use_fixed_grid ? time_grid.back() : end_time);
  essvec.push_back(essw(normw));

  Rcpp::List result;
  result.push_back(out, "out");
  result.push_back(normw, "normw");
  result.push_back(time_mesh, "time_mesh");
  result.push_back(essvec, "essvec");
  // PHI component estimates use time_mesh without its initial entry.
  result.push_back(phi_gradient_component_estimate, "phi_gradient_component_estimate");
  result.push_back(phi_hessian_component_estimate, "phi_hessian_component_estimate");
  result.push_back(phi_gradient_component_unweighted_estimate_q01, "phi_gradient_component_unweighted_estimate_q01");
  result.push_back(phi_gradient_component_unweighted_estimate_q99, "phi_gradient_component_unweighted_estimate_q99");
  result.push_back(phi_hessian_component_unweighted_estimate_q01, "phi_hessian_component_unweighted_estimate_q01");
  result.push_back(phi_hessian_component_unweighted_estimate_q99, "phi_hessian_component_unweighted_estimate_q99");
  result.push_back(phi_unweighted_estimate_q01, "phi_unweighted_estimate_q01");
  result.push_back(phi_unweighted_estimate_q99, "phi_unweighted_estimate_q99");
  result.push_back(phi_weighted_estimate_q01, "phi_weighted_estimate_q01");
  result.push_back(phi_weighted_estimate_q99, "phi_weighted_estimate_q99");
  result.push_back(phi_unweighted_estimate, "phi_unweighted_estimate");
  result.push_back(phi_weighted_estimate, "phi_weighted_estimate");
  result.push_back(time_before_step, "time_before_step");
  result.push_back(time_after_step, "time_after_step");
  result.push_back(delta_used_vec, "delta_used");
  result.push_back(logical_vector_from_ints(resampled_before_step_vec), "resampled_before_step");
  result.push_back(logical_vector_from_ints(resampled_before_next_step_vec), "resampled_before_next_step");
  result.push_back(conditional_ess_ratio, "conditional_ess_ratio");
  result.push_back(conditional_ess, "conditional_ess");
  result.push_back(logweight_variance, "logweight_variance");
  result.push_back(log_incremental_weight_sd, "log_incremental_weight_sd");
  result.push_back(log_incremental_weight_q01, "log_incremental_weight_q01");
  result.push_back(log_incremental_weight_q50, "log_incremental_weight_q50");
  result.push_back(log_incremental_weight_q99, "log_incremental_weight_q99");
  result.push_back(log_incremental_weight_q99_q01, "log_incremental_weight_q99_q01");
  result.push_back(v_from_variance, "v_from_variance");
  result.push_back(v_from_conditional_ess, "v_from_conditional_ess");
  result.push_back(delta_ess_raw, "delta_ess_raw");
  result.push_back(delta_ess_bounded, "delta_ess_bounded");
  result.push_back(delta_variance_raw, "delta_variance_raw");
  result.push_back(delta_variance_bounded, "delta_variance_bounded");
  result.push_back(asymptotic_expected_nu, "asymptotic_expected_nu");
  result.push_back(asymptotic_k3, "asymptotic_k3");
  result.push_back(asymptotic_k4, "asymptotic_k4");
  result.push_back(asymptotic_delta_term1, "asymptotic_delta_term1");
  result.push_back(asymptotic_delta_term2, "asymptotic_delta_term2");
  result.push_back(asymptotic_selected_term, "asymptotic_selected_term");
  result.push_back(delta_asymptotic_raw, "delta_asymptotic_raw");
  result.push_back(delta_asymptotic_bounded, "delta_asymptotic_bounded");
  result.push_back(auto_first_step ? NA_INTEGER : nchunks, "initial_nchunks");
  result.push_back(initial_step_size, "initial_step_size");
  result.push_back(completed_ordinary_steps, "n_ordinary_steps");
  result.push_back(completed_ordinary_steps + 1, "n_total_steps");
  result.push_back(target_one_step_ess, "target_one_step_ess");
  result.push_back(step_size_min_factor, "step_size_min_factor");
  result.push_back(step_size_max_factor, "step_size_max_factor");
  result.push_back(step_size_total_min, "step_size_total_min");
  result.push_back(resampcount, "resampcount");
  result.push_back(align_phi_on_resample, "align_phi_on_resample");
  result.push_back(model_name, "model");
  result.push_back(effective_n_threads, "n_threads");
  result.push_back(std::string(use_fixed_grid ? "offline_grid" : "online"), "step_size_mode");
  if (use_fixed_grid) {
    result.push_back(static_cast<int>(time_grid.size() - 1), "fixed_grid_n_steps");
  }
  return result;
}


// -------------------------------------------------------------------------
// end_T tuning for the rejection sampler.
//  1) sample rejection energies from the initial particle cube
//  2) bracket the target-acceptance root in end_time
//  3) solve by bisection
// -------------------------------------------------------------------------

inline double rejection_acceptance_estimate_from_energies(const arma::vec &energies,
                                                          const double end_time) {
  // just a helper function, calculates the empirical acceptance 
  // probability (based on the energy e.g. -logU).
  // Was separated from the main function tune_rejection_end_time_cpp (below),
  // because it has to be calculate it several times.
  if (!std::isfinite(end_time) || end_time <= 0.0) {
    Rcpp::stop("end_time must be a positive finite number.");
  }

  const double inv_end_time = 1.0 / end_time;
  double acc_sum = 0.0;
  const double *ptr = energies.memptr(); // create a pointer to energy vector memory
  const arma::uword n = energies.n_elem; // get number of energies
  for (arma::uword i = 0; i < n; ++i) {
    acc_sum += std::exp(-ptr[i] * inv_end_time); // accumulate exponential weights
  }
  return acc_sum / static_cast<double>(n); // calculate 1/n \sum_i exp(-E_i/end_T)
}

inline bool build_child_weight_cdf(SEXP child_weights,
                                   const int nrow,
                                   const int nshards,
                                   arma::mat &cdf) {
  if (Rf_isNull(child_weights)) {
    return false;
  }
  if (!Rf_isMatrix(child_weights) || (!Rf_isReal(child_weights) && !Rf_isInteger(child_weights))) {
    Rcpp::stop("child_weights must be a numeric matrix.");
  }

  Rcpp::NumericMatrix weights = Rcpp::as<Rcpp::NumericMatrix>(child_weights);
  if (weights.nrow() != nrow || weights.ncol() != nshards) {
    Rcpp::stop("child_weights has incompatible dimensions.");
  }

  cdf.set_size(nrow, nshards);
  for (int c = 0; c < nshards; ++c) {
    double sum_w = 0.0;
    for (int i = 0; i < nrow; ++i) {
      const double w = weights(i, c);
      if (!std::isfinite(w)) {
        Rcpp::stop("child_weights must contain only finite values.");
      }
      if (w < 0.0) {
        Rcpp::stop("child_weights must be nonnegative.");
      }
      sum_w += w;
    }
    if (!(sum_w > 0.0)) {
      Rcpp::stop("child_weights columns must have positive sum.");
    }
    if (std::abs(sum_w - 1.0) > 1e-10) {
      Rcpp::stop("child_weights columns must sum to one within 1e-10.");
    }

    double cumulative = 0.0;
    for (int i = 0; i < nrow; ++i) {
      cumulative += weights(i, c) / sum_w;
      cdf(i, c) = cumulative;
    }
    cdf(nrow - 1, c) = 1.0;
  }

  return true;
}

inline int sample_child_index_from_cdf(const arma::mat &cdf, const int child) {
  double u = R::runif(0.0, 1.0);
  if (u >= 1.0) {
    u = std::nextafter(1.0, 0.0);
  }

  const double *begin = cdf.colptr(child);
  const double *end = begin + cdf.n_rows;
  const double *it = std::upper_bound(begin, end, u);
  if (it == end) {
    Rcpp::stop("child_weights CDF sampling failed.");
  }
  return static_cast<int>(it - begin);
}

inline arma::imat sample_indices_from_child_cdfs(const int n_samples,
                                                 const int nshards,
                                                 const arma::mat &cdf) {
  arma::imat sampled_rows(n_samples, nshards);
  for (int i = 0; i < n_samples; ++i) {
    if (i % 10000 == 0) { Rcpp::checkUserInterrupt(); }
    for (int c = 0; c < nshards; ++c) {
      sampled_rows(i, c) = sample_child_index_from_cdf(cdf, c);
    }
  }
  return sampled_rows;
}

// The energy of a proposal reads the d coordinates of one particle per shard.
// In the (npart, nshards, ndim) particle cube those coordinates are strided by
// npart * nshards, so each one is a separate cache miss. The energy routines
// below therefore work on a transposed copy with dimensions
// (ndim, nshards, npart), in which the d coordinates of a given shard particle
// are contiguous. Building the copy costs O(npart * nshards * ndim) once,
// against O(npart * nshards * ndim / acceptance) for the proposals themselves.
inline arma::cube transpose_particle_cube(const arma::cube &init_particles) {
  const int npart = static_cast<int>(init_particles.n_rows);
  const int nshards = static_cast<int>(init_particles.n_cols);
  const int ndim = static_cast<int>(init_particles.n_slices);

  arma::cube particles_t(ndim, nshards, npart);
  for (int d = 0; d < ndim; ++d) {
    for (int c = 0; c < nshards; ++c) {
      for (int i = 0; i < npart; ++i) {
        particles_t(d, c, i) = init_particles(i, c, d);
      }
    }
  }
  return particles_t;
}

// Start of the contiguous ndim-vector of shard c of particle row.
inline const double * particle_ptr(const arma::cube &particles_t,
                                   const int row,
                                   const int shard) {
  const std::size_t offset =
    (static_cast<std::size_t>(row) * particles_t.n_cols + shard) * particles_t.n_rows;
  return particles_t.memptr() + offset;
}

inline arma::vec calc_energies_for_indices(const arma::cube &particles_t,
                                           const arma::imat &sampled_rows) {
  const int n_samples = static_cast<int>(sampled_rows.n_rows);
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  arma::vec energies(n_samples, arma::fill::zeros);

  #pragma omp parallel
  {
  arma::rowvec shard_means(ndim, arma::fill::zeros);
  std::vector<const double *> theta(nshards, nullptr);
  #pragma omp for schedule(static)
  for (int i = 0; i < n_samples; ++i) {
    shard_means.zeros();
    for (int c = 0; c < nshards; ++c) {
      theta[c] = particle_ptr(particles_t, sampled_rows(i, c), c);
    }

    for (int d = 0; d < ndim; ++d) {
      double mean_d = 0.0;
      for (int c = 0; c < nshards; ++c) {
        mean_d += theta[c][d];
      }
      shard_means[d] = mean_d / static_cast<double>(nshards);
    }

    double energy = 0.0;
    for (int d = 0; d < ndim; ++d) {
      const double mean_d = shard_means[d];
      for (int c = 0; c < nshards; ++c) {
        const double diff = theta[c][d] - mean_d;
        energy += diff * diff;
      }
    }
    energies[i] = 0.5 * energy;
  }
  }

  return energies;
}

inline arma::vec calc_energies_for_indices_diagonal(const arma::cube &particles_t,
                                                    const arma::imat &sampled_rows,
                                                    const AbfPreconditioner &precond) {
  const int n_samples = static_cast<int>(sampled_rows.n_rows);
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  arma::vec energies(n_samples, arma::fill::zeros);

  #pragma omp parallel
  {
  arma::rowvec centre(ndim, arma::fill::zeros);
  std::vector<const double *> theta(nshards, nullptr);
  #pragma omp for schedule(static)
  for (int i = 0; i < n_samples; ++i) {
    centre.zeros();
    for (int c = 0; c < nshards; ++c) {
      theta[c] = particle_ptr(particles_t, sampled_rows(i, c), c);
    }

    for (int d = 0; d < ndim; ++d) {
      double weighted_sum = 0.0;
      for (int c = 0; c < nshards; ++c) {
        weighted_sum += precond.q_inv_diag(c, d) * theta[c][d];
      }
      centre[d] = precond.q_star_diag[d] * weighted_sum;
    }

    double energy = 0.0;
    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        const double diff = theta[c][d] - centre[d];
        energy += precond.q_inv_diag(c, d) * diff * diff;
      }
    }
    energies[i] = 0.5 * energy;
  }
  }

  return energies;
}

inline arma::vec calc_energies_for_indices_full(const arma::cube &particles_t,
                                                const arma::imat &sampled_rows,
                                                const AbfPreconditioner &precond) {
  const int n_samples = static_cast<int>(sampled_rows.n_rows);
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  arma::vec energies(n_samples, arma::fill::zeros);

  #pragma omp parallel
  {
  arma::rowvec weighted_sum(ndim, arma::fill::zeros);
  arma::rowvec centre(ndim, arma::fill::zeros);
  arma::rowvec diff(ndim, arma::fill::zeros);
  #pragma omp for schedule(static)
  for (int i = 0; i < n_samples; ++i) {
    weighted_sum.zeros();
    for (int c = 0; c < nshards; ++c) {
      // non-owning view of the contiguous coordinates, so no gather is needed
      const arma::rowvec theta(const_cast<double *>(particle_ptr(particles_t, sampled_rows(i, c), c)),
                               ndim, false, true);
      weighted_sum += theta * precond.q_inv[c];
    }
    centre = weighted_sum * precond.q_star;

    double energy = 0.0;
    for (int c = 0; c < nshards; ++c) {
      const arma::rowvec theta(const_cast<double *>(particle_ptr(particles_t, sampled_rows(i, c), c)),
                               ndim, false, true);
      diff = theta - centre;
      energy += arma::as_scalar(diff * precond.q_inv[c] * diff.t());
    }
    energies[i] = 0.5 * energy;
  }
  }

  return energies;
}

inline arma::vec sample_particles_calc_energies(const arma::cube &particles_t, // sample_rejection_energies_cpp
                                               const int n_samples) {
  // samples n_samples independent energies from the transposed particle cube
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  const int npart = static_cast<int>(particles_t.n_slices);

  if (npart <= 0 || nshards <= 0 || ndim <= 0) {
    Rcpp::stop("init_particles must be a 3D array with positive dimensions.");
  }
  if (n_samples <= 0) {
    Rcpp::stop("n_samples must be a positive integer.");
  }

  arma::imat sampled_rows(n_samples, nshards);
  for (int i = 0; i < n_samples; ++i) {
    if (i % 10000 == 0) { Rcpp::checkUserInterrupt(); }
    for (int c = 0; c < nshards; ++c) { // draw an index from 1...N per shard
      sampled_rows(i, c) = static_cast<int>(npart * unif_rand());
    }
  }

  return calc_energies_for_indices(particles_t, sampled_rows);
}

inline arma::vec sample_particles_calc_energies_diagonal(const arma::cube &particles_t,
                                                         const int n_samples,
                                                         const AbfPreconditioner &precond) {
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  const int npart = static_cast<int>(particles_t.n_slices);

  if (npart <= 0 || nshards <= 0 || ndim <= 0) {
    Rcpp::stop("init_particles must be a 3D array with positive dimensions.");
  }
  if (n_samples <= 0) {
    Rcpp::stop("n_samples must be a positive integer.");
  }

  arma::imat sampled_rows(n_samples, nshards);
  for (int i = 0; i < n_samples; ++i) {
    if (i % 10000 == 0) { Rcpp::checkUserInterrupt(); }
    for (int c = 0; c < nshards; ++c) {
      sampled_rows(i, c) = static_cast<int>(npart * unif_rand());
    }
  }

  return calc_energies_for_indices_diagonal(particles_t, sampled_rows, precond);
}

inline arma::vec sample_particles_calc_energies_full(const arma::cube &particles_t,
                                                     const int n_samples,
                                                     const AbfPreconditioner &precond) {
  const int ndim = static_cast<int>(particles_t.n_rows);
  const int nshards = static_cast<int>(particles_t.n_cols);
  const int npart = static_cast<int>(particles_t.n_slices);

  if (npart <= 0 || nshards <= 0 || ndim <= 0) {
    Rcpp::stop("init_particles must be a 3D array with positive dimensions.");
  }
  if (n_samples <= 0) {
    Rcpp::stop("n_samples must be a positive integer.");
  }

  arma::imat sampled_rows(n_samples, nshards);
  for (int i = 0; i < n_samples; ++i) {
    if (i % 10000 == 0) { Rcpp::checkUserInterrupt(); }
    for (int c = 0; c < nshards; ++c) {
      sampled_rows(i, c) = static_cast<int>(npart * unif_rand());
    }
  }

  return calc_energies_for_indices_full(particles_t, sampled_rows, precond);
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List tune_rejection_end_time_cpp(const arma::cube &init_particles,
                                       const int n_samples,
                                       const double target_acceptance,
                                       const double lower = 1e-16,
                                       Rcpp::Nullable<double> upper = R_NilValue, // unused atm, pick adaptively in this function
                                       Rcpp::Nullable<double> tol = R_NilValue,
                                       const double expand_factor = 2.0,
                                       const int max_expand = 50,
                                       const std::string preconditioning = "no",
                                       SEXP abf_preconditioner = R_NilValue,
                                       SEXP child_weights = R_NilValue,
                                       int n_threads = 1) {
  // find the end_T that gives the desired acceptance from the rejection sampler.
  
  Rcpp::RNGScope scope;
  
  // draw a combination of particles, one for each shard, and calc energies
  // estimate time as a function of acceptance. Then calc distance from tgt acceptance.

  // safety checks
  const int effective_n_threads = apply_openmp_thread_count(n_threads);
  if (!std::isfinite(target_acceptance) || target_acceptance <= 0.0 || target_acceptance >= 1.0) {
    Rcpp::stop("target_acceptance must lie in (0, 1).");
  }
  if (!std::isfinite(lower) || lower <= 0.0) {
    Rcpp::stop("lower must be a positive finite number.");
  }
  if (!std::isfinite(expand_factor) || expand_factor <= 1.0) {
    Rcpp::stop("expand_factor must be greater than 1.");
  }
  if (max_expand < 1) {
    Rcpp::stop("max_expand must be a positive integer.");
  }
  if (n_samples <= 0) {
    Rcpp::stop("n_samples must be a positive integer.");
  }
  if (init_particles.n_rows < 1u || init_particles.n_slices < 1u) {
    Rcpp::stop("init_particles must be a 3D array with positive dimensions.");
  }
  if (init_particles.n_cols < 2u) {
    Rcpp::stop("tune_rejection_end_time_cpp requires at least two shards.");
  }
  arma::mat child_weight_cdf;
  const bool weighted_child_proposals = build_child_weight_cdf(
    child_weights,
    static_cast<int>(init_particles.n_rows),
    static_cast<int>(init_particles.n_cols),
    child_weight_cdf
  );

  const AbfPreconditioner precond = make_abf_preconditioner(
    preconditioning,
    abf_preconditioner,
    static_cast<int>(init_particles.n_cols),
    static_cast<int>(init_particles.n_slices)
  );

  const arma::cube particles_t = transpose_particle_cube(init_particles);

  // energies are sampled only once.
  // the root solve is deterministic conditional on the sampled energies.
  arma::vec energies;
  if (weighted_child_proposals) {
    arma::imat sampled_rows = sample_indices_from_child_cdfs(
      n_samples,
      static_cast<int>(init_particles.n_cols),
      child_weight_cdf
    );
    if (precond.mode == AbfPreconditioningMode::No) {
      energies = calc_energies_for_indices(particles_t, sampled_rows);
    } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
      energies = calc_energies_for_indices_diagonal(particles_t, sampled_rows, precond);
    } else {
      energies = calc_energies_for_indices_full(particles_t, sampled_rows, precond);
    }
  } else {
    if (precond.mode == AbfPreconditioningMode::No) {
      energies = sample_particles_calc_energies(particles_t, n_samples);
    } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
      energies = sample_particles_calc_energies_diagonal(particles_t, n_samples, precond);
    } else {
      energies = sample_particles_calc_energies_full(particles_t, n_samples, precond);
    }
  }

  // Build default upper and tol.
  const double mean_energy = arma::mean(energies);
  // if upper is passed, use that.
  double upper_value = upper.isNotNull()
    ? Rcpp::as<double>(upper)
    // if not, use mean_energy / std::max(1e-8, std::abs(std::log(target_acceptance))), with a lower bound of epsilon()
    : std::max(static_cast<double>(std::numeric_limits<double>::epsilon()),
               mean_energy / std::max(1e-8, std::abs(std::log(target_acceptance))));
  // pretend the energy is constant, then exp(-E/t) = a, then t = E / |log a|. max(1e-8) is against dividing by 0.
  double tol_value = tol.isNotNull()
    ? Rcpp::as<double>(tol)
    : std::max(1e-10, 1e-6 * target_acceptance);
  // The bisection stops when bracket_upper - bracket_lower <= tol_value, an
  // absolute tolerance on the end time.

  if (!std::isfinite(upper_value) || upper_value <= lower) {
    Rcpp::stop("upper must be a finite number greater than lower.");
  }
  if (!std::isfinite(tol_value) || tol_value <= 0.0) {
    Rcpp::stop("tol must be a positive finite number.");
  }

  // define root objective function
  auto objective = [&](const double end_time) {
    return rejection_acceptance_estimate_from_energies(energies, end_time) - target_acceptance;
  };

  // bracketing phase
  double bracket_lower = lower;
  double bracket_upper = upper_value;
  double f_lower = objective(bracket_lower);
  double f_upper = objective(bracket_upper);
  int n_expand = 0;

  // expand the bracket until the root is in it, or one endpoint hits zero. 
  // At zero you stop and return, because you found the root. A bit clunky but hey.
  while ((std::signbit(f_lower) == std::signbit(f_upper)) || f_lower == 0.0 || f_upper == 0.0) {
    if (f_lower == 0.0) {
      bracket_upper = bracket_lower;
      break;
    }
    if (f_upper == 0.0) {
      bracket_lower = bracket_upper;
      break;
    }
    if (n_expand >= max_expand) {
      Rcpp::stop("Failed to bracket a sign change.");
    }

    bracket_lower = bracket_upper;
    f_lower = f_upper;
    bracket_upper *= expand_factor;
    f_upper = objective(bracket_upper);
    ++n_expand;
  }

  // bisection solver
  while ((bracket_upper - bracket_lower) > tol_value) {
    const double mid = 0.5 * (bracket_lower + bracket_upper);
    const double f_mid = objective(mid);

    if (f_mid == 0.0) { // break if mid point is on root
      bracket_lower = mid;
      bracket_upper = mid;
      break;
    }

    // decide which half to keep.
    if (std::signbit(f_mid) == std::signbit(f_lower)) {
      bracket_lower = mid;
      f_lower = f_mid;
    } else {
      bracket_upper = mid;
      f_upper = f_mid;
    }
  }

  // final estimate
  const double end_time = 0.5 * (bracket_lower + bracket_upper);
  const double acceptance = rejection_acceptance_estimate_from_energies(energies, end_time);

  return Rcpp::List::create(
    Rcpp::_["end_time"] = end_time,
    Rcpp::_["target_acceptance"] = target_acceptance,
    Rcpp::_["acceptance"] = Rcpp::List::create(
      Rcpp::_["estimate"] = acceptance,
      Rcpp::_["n_energies"] = static_cast<int>(energies.n_elem)
    ),
    Rcpp::_["n_energies"] = static_cast<int>(energies.n_elem),
    Rcpp::_["energies"] = energies,
    Rcpp::_["n_expand"] = n_expand,
    Rcpp::_["weighted_child_proposals"] = weighted_child_proposals,
    Rcpp::_["n_threads"] = effective_n_threads
  );
}


// -------------------------------------------------------------------------
// Exported helpers used elsewhere in the R code.
// -------------------------------------------------------------------------


inline double rwlogit_softplus_sum(const arma::vec &eta,
                                   arma::vec &abs_eta_scratch) { // a scratch vector used as workspace
  // eta = X * beta. Computes sum_i log (1+e^{eta_i}) in a numerically stable way. 
  // Is used in the negative log likelihood (energy potential). Called by rwlogit_potential_from_eta
  // Uses the identity log(1+e^x) = max(x,0) + log(1 + e^{−∣x∣}) to avoid over and underflow in e^x.
  // i.e. in our case: sum_i log(1+e^{eta_i}) = sum_i [ max(eta_i,0) + log(1+e^{−∣eta_i∣}) ].
  // The scratch vector is reused so it doesn't need to be reallocated at every call.
  
  // calculate sum_i max(eta_i,0)
  // uses the identity max(x,0) = (x + |x|)/2
  abs_eta_scratch = arma::abs(eta);
  const double sum_positive = 0.5 * (arma::accu(eta) + arma::accu(abs_eta_scratch)); // calc positive part of the sum

  // calculate sum_i log(1+e^{−∣eta_i∣})
  const double *ptr = abs_eta_scratch.memptr(); // create pointer to absolute values. Used for speed.
  double tail_sum = 0.0;
  for (arma::uword i = 0; i < abs_eta_scratch.n_elem; ++i) {
    tail_sum += std::log1p(std::exp(-ptr[i])); // same as abs_eta_scratch[i] but faster.
  }

  return sum_positive + tail_sum;
}


inline double rwlogit_potential_from_eta(const arma::vec &Y,
                                         const arma::vec &eta,
                                         arma::vec &abs_eta_scratch) {
  // Calculate the logistic negative log-likelihood (energy potential) as a function of eta = X * beta.
  // Literally U(eta) = - log pi (eta).
  // I.e. sum_i log(1 + e^{eta_i}) − sum_i Y_i * eta_i .
  // Once again, the scratch vector is reused so it doesn't need to be reallocated at every call.
  // The prior is assumed to be \propto 1.
  
  return rwlogit_softplus_sum(eta, abs_eta_scratch) - arma::dot(Y, eta);
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List rwLogitChol(const arma::vec &Y,
                       const arma::mat &X,
                       arma::vec xi,
                       const double &sigma,
                       const int &run,
                       const arma::mat &chol_A) { // pre-computed, saves time for multiple runs.
  // Random-walk Metropolis sampler for logistic regression, 
  // using a Cholesky-preconditioned Gaussian proposal.
  
  Rcpp::RNGScope scope;

  // safety checks
  const int d = static_cast<int>(xi.n_elem);
  if (run <= 0) {
    Rcpp::stop("rwLogitChol: run must be positive.");
  }
  if (static_cast<int>(chol_A.n_rows) != d || static_cast<int>(chol_A.n_cols) != d) {
    Rcpp::stop("rwLogitChol: chol_A has incompatible dimensions.");
  }
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    Rcpp::stop("rwLogitChol: sigma must be finite and positive.");
  }

  // pre-allocate objects.
  arma::vec eta_current = X * xi;
  arma::vec eta_proposal(X.n_rows, arma::fill::zeros);
  arma::vec abs_eta_scratch(X.n_rows, arma::fill::zeros);
  double current_U = rwlogit_potential_from_eta(Y, eta_current, abs_eta_scratch);

  const arma::mat X_chol_A = X * chol_A;
  arma::vec z(d, arma::fill::zeros);
  arma::vec step_xi(d, arma::fill::zeros);
  arma::vec step_eta(X.n_rows, arma::fill::zeros);
  arma::mat out(d, run, arma::fill::zeros); // d x N output! it's transposed to store xi as a column vec, more efficient for Armadillo
  int accepted = 0;

  for (int n = 0; n < run; ++n) {
    if (n % 10000 == 0) { Rcpp::checkUserInterrupt(); }

    z.randn();
    step_xi = chol_A * z; // xi' = xi + cholA*z
    step_eta = X_chol_A * z; // eta' = X*xi' = X*xi + sigma*X*cholA*z

    eta_proposal = eta_current + sigma * step_eta; // proposal in eta (predictor) space
    const double proposal_U = rwlogit_potential_from_eta(Y, eta_proposal, abs_eta_scratch);
    const double log_alpha = current_U - proposal_U;

    if (log_alpha >= 0.0 || std::log(R::runif(0.0, 1.0)) < log_alpha) {
      xi += sigma * step_xi;
      eta_current.swap(eta_proposal); // swap is more efficient than copy, and current gets overwritten later
      current_U = proposal_U; // this is just a double.
      ++accepted;
    }

    out.col(n) = xi; //out.row(n) = xi.t();
  }

  return Rcpp::List::create(Rcpp::_["samples"] = out.t(), // transpose out to be Nxd!
                      Rcpp::_["acc_rate"] = static_cast<double>(accepted) / static_cast<double>(run),
                      Rcpp::_["last_state"] = xi); // to continue from, for a warm start next time. Slightly redundant.
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List rwLogitTuneChol(const arma::vec &Y,
                           const arma::mat &X,
                           arma::vec xi,
                           const double &sigma_init,
                           const int &batch_size,
                           const int &n_batches,
                           const arma::mat &chol_A, // this stays fixed through the run.
                           const double &target_acc = 0.35,
                           const double &gamma0 = 0.05,
                           const double &gamma_decay = 0.5) {
  // This function does adaptive tuning of the random-walk scale sigma for the logit MH kernel.
  // Does not return a finished MCMC sample (saves on allocating and returning arma::mat out).
  // Run a Cholesky-preconditioned random-walk Metropolis chain
  // monitor acceptance rate in batches
  // adjust sigma after each batch so acceptance moves toward target_acc
  // return the tuned final sigma plus some diagnostics.
  // Inputs:
  //   Y, X: logistic regression data
  //   xi: starting parameter vector
  //   sigma_init: initial proposal scale
  //   batch_size: number of MH iterations before each scale update
  //   n_batches: how many adaptation rounds to do
  //   chol_A: Cholesky factor shaping the proposal
  //   target_acc: desired batch acceptance rate, default 0.35
  //   gamma0, gamma_decay: learning-rate parameters controlling how aggressively sigma is updated
  //   gamma0: initial aggressiveness of tuning
  //   gamma_decay: how fast that aggressiveness shrinks as the batches progress.
  //   learning rate: gamma = gamma0 / (1 + b)^gamma_decay
  
  Rcpp::RNGScope scope;

  const int d = static_cast<int>(xi.n_elem);
  
  // safety checks
  if (static_cast<int>(chol_A.n_rows) != d || static_cast<int>(chol_A.n_cols) != d) {
    Rcpp::stop("rwLogitTuneChol: chol_A has incompatible dimensions.");
  }
  if (batch_size <= 0 || n_batches <= 0) {
    Rcpp::stop("rwLogitTuneChol: batch_size and n_batches must be positive.");
  }
  if (!std::isfinite(sigma_init) || sigma_init <= 0.0) {
    Rcpp::stop("rwLogitTuneChol: sigma_init must be finite and positive.");
  }

  // allocate objects and pre-compute
  arma::vec eta_current = X * xi;
  arma::vec eta_proposal(X.n_rows, arma::fill::zeros);
  arma::vec abs_eta_scratch(X.n_rows, arma::fill::zeros);
  double current_U = rwlogit_potential_from_eta(Y, eta_current, abs_eta_scratch);

  const arma::mat X_chol_A = X * chol_A;
  arma::vec z(d, arma::fill::zeros);
  arma::vec step_xi(d, arma::fill::zeros);
  arma::vec step_eta(X.n_rows, arma::fill::zeros);

  arma::vec batch_acceptance(n_batches, arma::fill::zeros);

  int accepted_total = 0;
  double log_sigma = std::log(sigma_init); // log_sigma is the adapted quantity
  
  for (int b = 0; b < n_batches; ++b) { // outer loop on batches
    if (b % 50 == 0) { Rcpp::checkUserInterrupt(); }

    const double sigma = std::exp(log_sigma); // sigma fixed within each batch
    int accepted_batch = 0;

    for (int t = 0; t < batch_size; ++t) { // inner loop within batches
      z.randn();
      step_xi = chol_A * z;
      step_eta = X_chol_A * z;

      eta_proposal = eta_current + sigma * step_eta;
      const double proposal_U = rwlogit_potential_from_eta(Y, eta_proposal, abs_eta_scratch);
      const double log_alpha = current_U - proposal_U;

      if (log_alpha >= 0.0 || std::log(R::runif(0.0, 1.0)) < log_alpha) {
        xi += sigma * step_xi;
        eta_current.swap(eta_proposal);
        current_U = proposal_U;
        ++accepted_batch;
        ++accepted_total;
      }
    }

    const double acc_batch = static_cast<double>(accepted_batch) / static_cast<double>(batch_size);
    batch_acceptance[b] = acc_batch;

    // adapt sigma
    const double gamma = gamma0 / std::pow(1.0 + static_cast<double>(b), gamma_decay);
    log_sigma += gamma * (acc_batch - target_acc); // gamma: learning rate
  }

  return Rcpp::List::create(Rcpp::_["sigma_init"] = sigma_init,
                      Rcpp::_["sigma_final"] = std::exp(log_sigma),
                      Rcpp::_["batch_acceptance"] = batch_acceptance,
                      Rcpp::_["overall_acc_rate"] = static_cast<double>(accepted_total) / static_cast<double>(batch_size * n_batches),
                      Rcpp::_["last_state"] = xi,
                      Rcpp::_["target_acc"] = target_acc,
                      Rcpp::_["final_acc"] = batch_acceptance[n_batches - 1],
                      Rcpp::_["mean_acc"] = arma::mean(batch_acceptance));
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::cube rej_samp_init_part(int out_length_tot,
                               double end_time,
                               const arma::cube &init_part,
                               double part_untouched = 0.0,
                               int chunk_size = 1000,
                               long long max_iter = 1e8,
                               const std::string preconditioning = "no",
                               SEXP abf_preconditioner = R_NilValue,
                               SEXP child_weights = R_NilValue,
                               int n_threads = 1) {
  // this is the actual (batched) rejection sampler.

  Rcpp::RNGScope scope;

  const int effective_n_threads = apply_openmp_thread_count(n_threads);
  (void) effective_n_threads;

  if (out_length_tot < 1) {
    Rcpp::stop("out_length_tot must be a positive integer.");
  }
  if (!std::isfinite(end_time) || end_time <= 0.0) {
    Rcpp::stop("end_time must be a positive finite number.");
  }
  if (!std::isfinite(part_untouched) || part_untouched < 0.0 || part_untouched > 1.0) {
    Rcpp::stop("part_untouched must lie in [0, 1].");
  }
  if (chunk_size < 1) {
    Rcpp::stop("chunk_size must be a positive integer.");
  }
  if (max_iter < 1) {
    Rcpp::stop("max_iter must be a positive integer.");
  }

  const int nrow = static_cast<int>(init_part.n_rows);
  const int nshards = static_cast<int>(init_part.n_cols);
  const int ndim = static_cast<int>(init_part.n_slices);
  if (nrow < 1 || nshards < 1 || ndim < 1) {
    Rcpp::stop("init_part must be a 3D particle cube with positive dimensions.");
  }
  if (!Rf_isNull(child_weights) && part_untouched > 0.0) {
    Rcpp::stop("child_weights cannot be supplied when part_untouched > 0.");
  }
  arma::mat child_weight_cdf;
  const bool weighted_child_proposals = build_child_weight_cdf(
    child_weights,
    nrow,
    nshards,
    child_weight_cdf
  );

  const AbfPreconditioner precond = make_abf_preconditioner(
    preconditioning,
    abf_preconditioner,
    nshards,
    ndim
  );

  // allocate memory and pre-compute objects
  const double inv_end_time = 1.0 / end_time;

  const arma::cube particles_t = transpose_particle_cube(init_part);

  arma::cube out(out_length_tot, nshards, ndim);

  const int out_rej = static_cast<int>(out_length_tot * (1.0 - part_untouched));

  long long accepted = 0;
  long long n_tested = 0;

  arma::imat indices(chunk_size, nshards);
  arma::vec mlogAcc(chunk_size);
  arma::vec rexpVals(chunk_size);

  while (accepted < out_rej && n_tested < max_iter) {
    if (weighted_child_proposals) {
      indices = sample_indices_from_child_cdfs(chunk_size, nshards, child_weight_cdf);
    } else {
      indices = arma::randi<arma::imat>(chunk_size, nshards, arma::distr_param(0, nrow - 1)); // chunk_size x C matrix
    }
    for (int i = 0; i < chunk_size; ++i) {
      rexpVals[i] = -std::log(R::runif(0.0, 1.0));
    }

    arma::vec batch_energies;
    if (precond.mode == AbfPreconditioningMode::No) {
      batch_energies = calc_energies_for_indices(particles_t, indices);
    } else if (precond.mode == AbfPreconditioningMode::Diagonal) {
      batch_energies = calc_energies_for_indices_diagonal(particles_t, indices, precond);
    } else {
      batch_energies = calc_energies_for_indices_full(particles_t, indices, precond);
    }
    mlogAcc = batch_energies * inv_end_time;

    // store those that pass the acceptance threshold
    arma::uvec keep = arma::find(rexpVals > mlogAcc);
    if (!keep.is_empty()) {
      const int ncopy = std::min<int>(keep.n_elem, out_rej - accepted);
      keep = keep.head(ncopy);

      // append to out, taking the accepted proposals straight from init_part
      for (int d = 0; d < ndim; ++d) {
        for (int c = 0; c < nshards; ++c) {
          for (int k = 0; k < ncopy; ++k) {
            out(accepted + k, c, d) = init_part(indices(keep[k], c), c, d);
          }
        }
      }
      accepted += ncopy;
    }

    n_tested += chunk_size;
  }

  if (accepted < out_rej) {
    Rcpp::stop(
      "rej_samp_init_part failed: accepted " +
        std::to_string(accepted) +
        " of " +
        std::to_string(out_rej) +
        " required rejection-sampled particles after testing " +
        std::to_string(n_tested) +
        " proposals. Increase max_iter or check end_time / target_acceptance."
    );
  }

  // Fill only the deliberately untouched fraction.
  // This is not a fallback for failed rejection sampling.
  const int untouched = out_length_tot - out_rej;
  if (untouched > 0) {
    arma::imat idx2 = arma::randi<arma::imat>(
      untouched,
      nshards,
      arma::distr_param(0, nrow - 1)
    );

    for (int d = 0; d < ndim; ++d) {
      for (int c = 0; c < nshards; ++c) {
        for (int i = 0; i < untouched; ++i) {
          out(out_rej + i, c, d) = init_part(idx2(i, c), c, d);
        }
      }
    }
  }

  return out;
}


// -------------------------------------------------------------------------
// Main C++ entry point used from R.
// No R phi callback is used in the SMC loop.
// -------------------------------------------------------------------------

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List bayesian_fusion_smc(double end_time,
                                SEXP nchunks,
                                std::string model,
                                arma::cube &init_particles,
                                Rcpp::List &params,
                                SEXP &init_weight,
                                bool resamp,
                                double resamp_thr,
                                bool only_one_step = false,
                                bool align_phi_on_resample = true,
                                std::string preconditioning = "no",
                                SEXP abf_preconditioner = R_NilValue,
                                double target_one_step_ess = 0.8,
                                double step_size_min_factor = 1.0,
                                double step_size_max_factor = 1.0,
                                double step_size_total_min = 0.0,
                                SEXP subposterior_means = R_NilValue,
                                int n_threads = 1,
                                std::string step_size_criterion = "ess_ratio",
                                SEXP time_grid = R_NilValue,
                                SEXP subposterior_variance = R_NilValue) {
  // Dispatcher: builds the model object named by model and runs the SMC.

  if (n_threads < 1) {
    Rcpp::stop("n_threads must be a positive integer.");
  }
  if (n_threads > 1 && model == "logit") {
    Rcpp::stop("n_threads > 1 is not supported for model == 'logit'.");
  }
  if (step_size_criterion != "ess_ratio" &&
      step_size_criterion != "asymptotic") {
    Rcpp::stop("step_size_criterion must be 'ess_ratio' or 'asymptotic'.");
  }

  if (!std::isfinite(end_time) || end_time <= 0.0) {
    Rcpp::stop("end_time must be a positive finite scalar.");
  }
  // nchunks = NULL asks for the first step to be taken from the spread of phi
  // across the initial particles rather than from a given number of chunks.
  const bool auto_first_step = Rf_isNull(nchunks);
  int nchunks_int = 0;
  if (!auto_first_step) {
    const double nchunks_value = Rcpp::as<double>(nchunks);
    if (!std::isfinite(nchunks_value) || nchunks_value <= 0.0 ||
        std::floor(nchunks_value) != nchunks_value ||
        nchunks_value > static_cast<double>(std::numeric_limits<int>::max())) {
      Rcpp::stop("nchunks must be a positive integer no larger than .Machine$integer.max, or NULL.");
    }
    nchunks_int = static_cast<int>(nchunks_value);
  }

  const bool use_fixed_grid = !Rf_isNull(time_grid);
  std::vector<double> validated_time_grid;
  if (use_fixed_grid) {
    if (only_one_step) {
      Rcpp::stop("time_grid cannot be supplied when only_one_step is TRUE.");
    }
    validated_time_grid = validate_time_grid(time_grid, end_time);
  }

  const int effective_n_threads = apply_openmp_thread_count(n_threads);
  validate_step_size_inputs(target_one_step_ess, step_size_min_factor, step_size_max_factor, step_size_total_min);

  const AbfPreconditioner precond = make_abf_preconditioner(
    preconditioning,
    abf_preconditioner,
    static_cast<int>(init_particles.n_cols),
    static_cast<int>(init_particles.n_slices)
  );
  const arma::mat subposterior_means_mat = make_subposterior_means(subposterior_means, init_particles);
  const double subposterior_variance_value =
    make_subposterior_variance(subposterior_variance, init_particles, precond);

  if (model == "rosen5d") {
    const Rosen5d model_object(params);
    return run_bayesian_fusion_smc(model_object, model, end_time, nchunks_int,
                                   init_particles, init_weight,
                                   resamp, resamp_thr,
                                   only_one_step, align_phi_on_resample, precond,
                                   target_one_step_ess, step_size_min_factor,
                                   step_size_max_factor, step_size_total_min,
                                   step_size_criterion,
                                   subposterior_means_mat,
                                   subposterior_variance_value,
                                   effective_n_threads,
                                   validated_time_grid, use_fixed_grid, auto_first_step);
  }
  if (model == "logit") {
    const Logit model_object(params);
    return run_bayesian_fusion_smc(model_object, model, end_time, nchunks_int,
                                   init_particles, init_weight,
                                   resamp, resamp_thr,
                                   only_one_step, align_phi_on_resample, precond,
                                   target_one_step_ess, step_size_min_factor,
                                   step_size_max_factor, step_size_total_min,
                                   step_size_criterion,
                                   subposterior_means_mat,
                                   subposterior_variance_value,
                                   effective_n_threads,
                                   validated_time_grid, use_fixed_grid, auto_first_step);
  }
  if (model == "logit_wprior_countsxypairs") {
    const Logit_wprior_countsxypairs model_object(params);
    return run_bayesian_fusion_smc(model_object, model, end_time, nchunks_int,
                                   init_particles, init_weight,
                                   resamp, resamp_thr,
                                   only_one_step, align_phi_on_resample, precond,
                                   target_one_step_ess, step_size_min_factor,
                                   step_size_max_factor, step_size_total_min,
                                   step_size_criterion,
                                   subposterior_means_mat,
                                   subposterior_variance_value,
                                   effective_n_threads,
                                   validated_time_grid, use_fixed_grid, auto_first_step);
  }
  if (model == "phbackblaze_centered_grouped_halfnormal") {
    const PHBackblaze_centered_grouped_halfnormal model_object(params);
    return run_bayesian_fusion_smc(model_object, model, end_time, nchunks_int,
                                   init_particles, init_weight,
                                   resamp, resamp_thr,
                                   only_one_step, align_phi_on_resample, precond,
                                   target_one_step_ess, step_size_min_factor,
                                   step_size_max_factor, step_size_total_min,
                                   step_size_criterion,
                                   subposterior_means_mat,
                                   subposterior_variance_value,
                                   effective_n_threads,
                                   validated_time_grid, use_fixed_grid, auto_first_step);
  }

  Rcpp::stop("Unknown model. Supported values are 'rosen5d', 'logit', 'logit_wprior_countsxypairs', and 'phbackblaze_centered_grouped_halfnormal'.");
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List bayesian_fusion_smc_lbb_poisson(double end_time,
                                           int nchunks,
                                           std::string model,
                                           arma::cube &init_particles,
                                           Rcpp::List &params,
                                           SEXP &init_weight,
                                           bool resamp,
                                           double resamp_thr,
                                           bool only_one_step = false,
                                           bool align_phi_on_resample = true) {
  Rcpp::RNGScope scope;
  // align_phi_on_resample is ignored because exact weights do not cache endpoint phi values.

  if (model != "logit") {
    Rcpp::stop("bayesian_fusion_smc_lbb_poisson supports only model == 'logit'.");
  }

  const Logit model_object(params);
  const Rcpp::Environment layeredBB = Rcpp::Environment::namespace_env("layeredBB");
  const Rcpp::Function multi_bessel_layer_simulation = layeredBB["multi_bessel_layer_simulation"];
  const Rcpp::Function multi_layered_brownian_bridge = layeredBB["multi_layered_brownian_bridge"];

  const int npart = static_cast<int>(init_particles.n_rows);
  const int nshards = static_cast<int>(init_particles.n_cols);
  const int ndim = static_cast<int>(init_particles.n_slices);

  int resampcount = 0;
  int lbb_segment_count = 0;
  int lbb_kap_max = 0;
  double lbb_bound_intensity_sum = 0.0;
  double lbb_bound_intensity_max = 0.0;
  double lbb_kap_sum = 0.0;

  arma::vec essvec(nchunks + 1, arma::fill::zeros);
  arma::vec cessvec(nchunks);
  arma::vec log_incremental_weight_sd(nchunks);
  arma::vec log_incremental_weight_q01(nchunks);
  arma::vec log_incremental_weight_q50(nchunks);
  arma::vec log_incremental_weight_q99(nchunks);
  arma::vec log_incremental_weight_q99_q01(nchunks);
  cessvec.fill(arma::datum::nan);
  log_incremental_weight_sd.fill(arma::datum::nan);
  log_incremental_weight_q01.fill(arma::datum::nan);
  log_incremental_weight_q50.fill(arma::datum::nan);
  log_incremental_weight_q99.fill(arma::datum::nan);
  log_incremental_weight_q99_q01.fill(arma::datum::nan);
  arma::vec nlogw(npart, arma::fill::zeros);
  arma::vec normw(npart, arma::fill::zeros);
  arma::cube x_old = init_particles;
  arma::cube x_new(npart, nshards, ndim, arma::fill::zeros);
  arma::mat x_bar(npart, ndim, arma::fill::zeros);
  arma::mat out(npart, ndim, arma::fill::zeros);
  arma::mat bridge_psi(npart, nshards, arma::fill::zeros);
  arma::vec bridge_chi(npart, arma::fill::zeros);
  arma::vec bridge_shift(npart, arma::fill::zeros);
  arma::vec x0(ndim, arma::fill::zeros);
  arma::vec x1(ndim, arma::fill::zeros);
  arma::vec log_incremental_weights(npart, arma::fill::zeros);

  compute_shard_means(x_old, x_bar);
  set_initial_weights(x_old, x_bar, end_time, init_weight, nlogw, normw);

  const double delta_t = end_time / static_cast<double>(nchunks);
  double previous_t = 0.0;
  double current_t = delta_t;

  double ess = essw(normw);
  essvec[0] = ess;

  for (int step = 1; step <= nchunks; ++step) {
    Rcpp::checkUserInterrupt();

    if (resamp && ess < static_cast<double>(npart) * resamp_thr) {
      ++resampcount;
      const arma::uvec idx = systematic_residual_resampling(normw);

      update_part_after_resamp(x_old, idx, x_new);
      x_old.swap(x_new);

      normw.fill(1.0 / static_cast<double>(npart));
      nlogw = -arma::log(normw);
    }

    if (step == nchunks) {
      break;
    }

    const double alpha = (end_time - current_t) / (end_time - previous_t);
    const double beta = delta_t / (end_time - previous_t);

    propagate_bridge(x_old, x_new, x_bar, bridge_psi, bridge_chi, bridge_shift, alpha, beta, delta_t);

    for (int i = 0; i < npart; ++i) {
      double segment_log_weight = 0.0;
      for (int c = 0; c < nshards; ++c) {
        for (int d = 0; d < ndim; ++d) {
          x0[d] = x_old(i, c, d);
          x1[d] = x_new(i, c, d);
        }

        const LbbPoissonSegmentWeight lbb = lbb_poisson_segment_weight(
            model_object,
            multi_bessel_layer_simulation,
            multi_layered_brownian_bridge,
            x0,
            x1,
            previous_t,
            current_t,
            c + 1,
            i + 1,
            step);

        segment_log_weight += lbb.log_weight;
        ++lbb_segment_count;
        lbb_bound_intensity_sum += lbb.bound_intensity;
        if (lbb.bound_intensity > lbb_bound_intensity_max) {
          lbb_bound_intensity_max = lbb.bound_intensity;
        }
        lbb_kap_sum += static_cast<double>(lbb.kap);
        if (lbb.kap > lbb_kap_max) {
          lbb_kap_max = lbb.kap;
        }
      }
      log_incremental_weights[i] = segment_log_weight;
      nlogw[i] -= segment_log_weight;
    }

    record_incremental_weight_diagnostics(
        normw, log_incremental_weights, step - 1, cessvec,
        log_incremental_weight_sd, log_incremental_weight_q01,
        log_incremental_weight_q50, log_incremental_weight_q99,
        log_incremental_weight_q99_q01);

    const arma::vec logw = -nlogw;
    bool any_finite_logw = false;
    for (int i = 0; i < npart; ++i) {
      if (std::isnan(logw[i])) {
        Rcpp::stop("bayesian_fusion_smc_lbb_poisson: exact log weight is NaN before normalization.");
      }
      if (std::isinf(logw[i]) && logw[i] > 0.0) {
        Rcpp::stop("bayesian_fusion_smc_lbb_poisson: exact log weight is +Inf before normalization.");
      }
      if (std::isfinite(logw[i])) {
        any_finite_logw = true;
      }
    }
    if (!any_finite_logw) {
      Rcpp::stop("bayesian_fusion_smc_lbb_poisson: all exact log weights are -Inf before normalization.");
    }
    normw = calc_weights(logw);

    x_old.swap(x_new);

    previous_t = current_t;
    current_t += delta_t;

    ess = essw(normw);
    essvec[step] = ess;

    if (only_one_step) {
      arma::mat out_one(npart, ndim, arma::fill::zeros);
      compute_shard_means(x_old, x_bar);
      out_one = x_bar;
      const double lbb_segment_count_d = static_cast<double>(lbb_segment_count);
      return Rcpp::List::create(Rcpp::_["out"] = out_one,
                          Rcpp::_["normw"] = normw,
                          Rcpp::_["end_time"] = end_time,
                          Rcpp::_["essvec"] = essvec,
                          Rcpp::_["cessvec"] = cessvec,
                          Rcpp::_["log_incremental_weight_sd"] = log_incremental_weight_sd,
                          Rcpp::_["log_incremental_weight_q01"] = log_incremental_weight_q01,
                          Rcpp::_["log_incremental_weight_q50"] = log_incremental_weight_q50,
                          Rcpp::_["log_incremental_weight_q99"] = log_incremental_weight_q99,
                          Rcpp::_["log_incremental_weight_q99_q01"] = log_incremental_weight_q99_q01,
                          Rcpp::_["resampcount"] = resampcount,
                          Rcpp::_["align_phi_on_resample"] = align_phi_on_resample,
                          Rcpp::_["model"] = model,
                          Rcpp::_["lbb_segment_count"] = lbb_segment_count,
                          Rcpp::_["lbb_bound_intensity_mean"] = lbb_bound_intensity_sum / lbb_segment_count_d,
                          Rcpp::_["lbb_bound_intensity_max"] = lbb_bound_intensity_max,
                          Rcpp::_["lbb_kap_mean"] = lbb_kap_sum / lbb_segment_count_d,
                          Rcpp::_["lbb_kap_max"] = lbb_kap_max);
    }
  }

  sample_fused_endpoint(x_old, x_bar, out, delta_t);

  for (int i = 0; i < npart; ++i) {
    double segment_log_weight = 0.0;
    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        x0[d] = x_old(i, c, d);
        x1[d] = out(i, d);
      }

      const LbbPoissonSegmentWeight lbb = lbb_poisson_segment_weight(
          model_object,
          multi_bessel_layer_simulation,
          multi_layered_brownian_bridge,
          x0,
          x1,
          previous_t,
          end_time,
          c + 1,
          i + 1,
          nchunks);

      segment_log_weight += lbb.log_weight;
      ++lbb_segment_count;
      lbb_bound_intensity_sum += lbb.bound_intensity;
      if (lbb.bound_intensity > lbb_bound_intensity_max) {
        lbb_bound_intensity_max = lbb.bound_intensity;
      }
      lbb_kap_sum += static_cast<double>(lbb.kap);
      if (lbb.kap > lbb_kap_max) {
        lbb_kap_max = lbb.kap;
      }
    }
    log_incremental_weights[i] = segment_log_weight;
    nlogw[i] -= segment_log_weight;
  }

  record_incremental_weight_diagnostics(
      normw, log_incremental_weights, nchunks - 1, cessvec,
      log_incremental_weight_sd, log_incremental_weight_q01,
      log_incremental_weight_q50, log_incremental_weight_q99,
      log_incremental_weight_q99_q01);

  const arma::vec logw = -nlogw;
  bool any_finite_logw = false;
  for (int i = 0; i < npart; ++i) {
    if (std::isnan(logw[i])) {
      Rcpp::stop("bayesian_fusion_smc_lbb_poisson: exact log weight is NaN before normalization.");
    }
    if (std::isinf(logw[i]) && logw[i] > 0.0) {
      Rcpp::stop("bayesian_fusion_smc_lbb_poisson: exact log weight is +Inf before normalization.");
    }
    if (std::isfinite(logw[i])) {
      any_finite_logw = true;
    }
  }
  if (!any_finite_logw) {
    Rcpp::stop("bayesian_fusion_smc_lbb_poisson: all exact log weights are -Inf before normalization.");
  }
  normw = calc_weights(logw);
  essvec[nchunks] = essw(normw);

  const double lbb_segment_count_d = static_cast<double>(lbb_segment_count);
  return Rcpp::List::create(Rcpp::_["out"] = out,
                      Rcpp::_["normw"] = normw,
                      Rcpp::_["essvec"] = essvec,
                      Rcpp::_["cessvec"] = cessvec,
                      Rcpp::_["log_incremental_weight_sd"] = log_incremental_weight_sd,
                      Rcpp::_["log_incremental_weight_q01"] = log_incremental_weight_q01,
                      Rcpp::_["log_incremental_weight_q50"] = log_incremental_weight_q50,
                      Rcpp::_["log_incremental_weight_q99"] = log_incremental_weight_q99,
                      Rcpp::_["log_incremental_weight_q99_q01"] = log_incremental_weight_q99_q01,
                      Rcpp::_["resampcount"] = resampcount,
                      Rcpp::_["align_phi_on_resample"] = align_phi_on_resample,
                      Rcpp::_["model"] = model,
                      Rcpp::_["lbb_segment_count"] = lbb_segment_count,
                      Rcpp::_["lbb_bound_intensity_mean"] = lbb_bound_intensity_sum / lbb_segment_count_d,
                      Rcpp::_["lbb_bound_intensity_max"] = lbb_bound_intensity_max,
                      Rcpp::_["lbb_kap_mean"] = lbb_kap_sum / lbb_segment_count_d,
                      Rcpp::_["lbb_kap_max"] = lbb_kap_max);
}
