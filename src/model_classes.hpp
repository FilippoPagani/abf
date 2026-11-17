// Model classes for the four examples of the paper. Each class stores the data
// of the C shards and computes, for a matrix of particle positions, the function
// phi of each shard, with no preconditioning, diagonal preconditioning or full
// preconditioning (the methods calc_phi_for_all_rows, ..._diagonal and
// ..._full). The other methods sum phi over the shards; utilitiesc.cpp calls
// them through calc_phi_cube_layer_sums_preconditioned and
// calc_phi_fused_last_step_preconditioned.
//   Rosen5d                                 5d hybrid Rosenbrock target
//   Logit                                   logistic regression, flat prior
//   Logit_wprior_countsxypairs              logistic regression, Gaussian
//                                           prior, counted (y, x) rows
//                                           (nycflights13)
//   PHBackblaze_centered_grouped_halfnormal Weibull proportional hazards
//                                           model (Backblaze)
#ifndef MODEL_CLASSES_HPP
#define MODEL_CLASSES_HPP

#include <RcppArmadillo.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

inline double logistic_stable(const double z) {
  if (z >= 0.0) {
    const double ez = std::exp(-z);
    return 1.0 / (1.0 + ez);
  }
  const double ez = std::exp(z);
  return ez / (1.0 + ez);
}

inline bool prepare_optional_phi_components(arma::vec *gradient_component,
                                            arma::vec *hessian_component,
                                            const int nrow,
                                            const char *caller) {
  if ((gradient_component == nullptr) != (hessian_component == nullptr)) {
    Rcpp::stop("%s expects both component outputs or neither.", caller);
  }
  if (gradient_component == nullptr) {
    return false;
  }
  if (static_cast<int>(gradient_component->n_elem) != nrow ||
      static_cast<int>(hessian_component->n_elem) != nrow) {
    Rcpp::stop("%s received component outputs with the wrong length.", caller);
  }
  gradient_component->zeros();
  hessian_component->zeros();
  return true;
}


class Rosen5d {
private:
  double a_;
  double b12_;
  double b13_;
  double b22_;
  double b23_;
  double mu_;


public:
  explicit Rosen5d(const Rcpp::List &params)
      : a_(Rcpp::as<double>(params[0])),
        b12_(Rcpp::as<double>(params[1])),
        b13_(Rcpp::as<double>(params[2])),
        b22_(Rcpp::as<double>(params[3])),
        b23_(Rcpp::as<double>(params[4])),
        mu_(Rcpp::as<double>(params[5])) {}

  inline double phi_rosen_5d(const double x1,
                           const double x12,
                           const double x13,
                           const double x22,
                           const double x23) const {
    // evaluate the phi function for a single observation from a 5d rosenbrock.
    const double x1_sq = x1 * x1;
    const double x12_sq = x12 * x12;
    const double x22_sq = x22 * x22;

    const double d12 = x12 - x1_sq;
    const double d13 = x13 - x12_sq;
    const double d22 = x22 - x1_sq;
    const double d23 = x23 - x22_sq;

    const double g1 = -2.0 * a_ * (x1 - mu_)
                    + 4.0 * b12_ * x1 * d12
                    + 4.0 * b22_ * x1 * d22;
    const double g2 = -2.0 * b12_ * d12
                    + 4.0 * b13_ * x12 * d13;
    const double g3 = -2.0 * b13_ * d13;
    const double g4 = -2.0 * b22_ * d22
                    + 4.0 * b23_ * x22 * d23;
    const double g5 = -2.0 * b23_ * d23;

    const double trace_hessian =
        (-2.0 * a_ + 4.0 * b12_ * (x12 - 3.0 * x1_sq) + 4.0 * b22_ * (x22 - 3.0 * x1_sq))
      + (-2.0 * b12_ + 4.0 * b13_ * (x13 - 3.0 * x12_sq))
      + (-2.0 * b13_)
      + (-2.0 * b22_ + 4.0 * b23_ * (x23 - 3.0 * x22_sq))
      + (-2.0 * b23_);

    return 0.5 * (g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4 + g5 * g5 + trace_hessian);
  }

  inline void rosen_5d_grad_hess(const double x1,
                                 const double x12,
                                 const double x13,
                                 const double x22,
                                 const double x23,
                                 arma::vec &grad,
                                 arma::vec &hdiag,
                                 double &h12,
                                 double &h14,
                                 double &h23,
                                 double &h45) const {
    if (grad.n_elem != 5u || hdiag.n_elem != 5u) {
      Rcpp::stop("rosen_5d_grad_hess expects length-5 scratch vectors.");
    }

    const double x1_sq = x1 * x1;
    const double x12_sq = x12 * x12;
    const double x22_sq = x22 * x22;

    const double d12 = x12 - x1_sq;
    const double d13 = x13 - x12_sq;
    const double d22 = x22 - x1_sq;
    const double d23 = x23 - x22_sq;

    grad[0] = -2.0 * a_ * (x1 - mu_)
            + 4.0 * b12_ * x1 * d12
            + 4.0 * b22_ * x1 * d22;
    grad[1] = -2.0 * b12_ * d12
            + 4.0 * b13_ * x12 * d13;
    grad[2] = -2.0 * b13_ * d13;
    grad[3] = -2.0 * b22_ * d22
            + 4.0 * b23_ * x22 * d23;
    grad[4] = -2.0 * b23_ * d23;

    hdiag[0] = -2.0 * a_ + 4.0 * b12_ * (x12 - 3.0 * x1_sq) + 4.0 * b22_ * (x22 - 3.0 * x1_sq);
    hdiag[1] = -2.0 * b12_ + 4.0 * b13_ * (x13 - 3.0 * x12_sq);
    hdiag[2] = -2.0 * b13_;
    hdiag[3] = -2.0 * b22_ + 4.0 * b23_ * (x23 - 3.0 * x22_sq);
    hdiag[4] = -2.0 * b23_;

    h12 = 4.0 * b12_ * x1;
    h14 = 4.0 * b22_ * x1;
    h23 = 4.0 * b13_ * x12;
    h45 = 4.0 * b23_ * x22;
  }

  inline void calc_phi_for_all_rows(const arma::mat &X,
                               arma::vec &out,
                               const int /*shard_unused*/,
                               arma::vec *gradient_component = nullptr,
                               arma::vec *hessian_component = nullptr) const {
    // Takes an N x d matrix of particle positions and stores phi at each row in
    // out. The shard index is 1-based, as in R. All shards have the same phi
    // here, so it is not used.
    if (X.n_cols != 5u) {
      Rcpp::stop("Rosen5d phi expects a matrix with exactly 5 columns.");
    }

    const int nrow = static_cast<int>(X.n_rows);
    if (static_cast<int>(out.n_elem) != nrow) {
      Rcpp::stop("calc_phi_for_all_rows: (out.n_elem) != nrow ");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, nrow, "Rosen5d::calc_phi_for_all_rows");

    const double *x1 = X.colptr(0);
    const double *x12 = X.colptr(1);
    const double *x13 = X.colptr(2);
    const double *x22 = X.colptr(3);
    const double *x23 = X.colptr(4);

    #pragma omp parallel
    {
    arma::vec grad(5, arma::fill::zeros);
    arma::vec hdiag(5, arma::fill::zeros);
    #pragma omp for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      out[i] = phi_rosen_5d(x1[i], x12[i], x13[i], x22[i], x23[i]);
      if (store_components) {
        double h12 = 0.0;
        double h14 = 0.0;
        double h23 = 0.0;
        double h45 = 0.0;
        rosen_5d_grad_hess(x1[i], x12[i], x13[i], x22[i], x23[i],
                           grad, hdiag, h12, h14, h23, h45);
        (*gradient_component)[i] = 0.5 * arma::dot(grad, grad);
        (*hessian_component)[i] = 0.5 * arma::accu(hdiag);
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_diagonal(const arma::mat &X,
                                             arma::vec &out,
                                             const int shard_1based,
                                             const arma::mat &q_diag,
                                             arma::vec *gradient_component = nullptr,
                                             arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= static_cast<int>(q_diag.n_rows)) {
      Rcpp::stop("Rosen5d received an invalid shard index.");
    }
    if (X.n_cols != 5u || q_diag.n_cols != 5u) {
      Rcpp::stop("Rosen5d diagonal phi expects five dimensions.");
    }

    const int nrow = static_cast<int>(X.n_rows);
    if (static_cast<int>(out.n_elem) != nrow) {
      Rcpp::stop("calc_phi_for_all_rows_diagonal: out.n_elem != nrow.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, nrow, "Rosen5d::calc_phi_for_all_rows_diagonal");

    #pragma omp parallel
    {
    arma::vec grad(5, arma::fill::zeros);
    arma::vec hdiag(5, arma::fill::zeros);
    #pragma omp for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      double h12 = 0.0;
      double h14 = 0.0;
      double h23 = 0.0;
      double h45 = 0.0;
      rosen_5d_grad_hess(X(i, 0), X(i, 1), X(i, 2), X(i, 3), X(i, 4),
                         grad, hdiag, h12, h14, h23, h45);
      double value = 0.0;
      double grad_quad = 0.0;
      double trace_term = 0.0;
      for (int d = 0; d < 5; ++d) {
        const double q_d = q_diag(shard_index, d);
        value += q_d * (grad[d] * grad[d] + hdiag[d]);
        grad_quad += q_d * grad[d] * grad[d];
        trace_term += q_d * hdiag[d];
      }
      out[i] = 0.5 * value;
      if (store_components) {
        (*gradient_component)[i] = 0.5 * grad_quad;
        (*hessian_component)[i] = 0.5 * trace_term;
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_full(const arma::mat &X,
                                         arma::vec &out,
                                         const int shard_1based,
                                         const std::vector<arma::mat> &q_cov,
                                         arma::vec *gradient_component = nullptr,
                                         arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= static_cast<int>(q_cov.size())) {
      Rcpp::stop("Rosen5d received an invalid shard index.");
    }
    const arma::mat &q_cov_c = q_cov[shard_index];
    if (X.n_cols != 5u || q_cov_c.n_rows != 5u || q_cov_c.n_cols != 5u) {
      Rcpp::stop("Rosen5d full phi expects five dimensions.");
    }

    const int nrow = static_cast<int>(X.n_rows);
    if (static_cast<int>(out.n_elem) != nrow) {
      Rcpp::stop("calc_phi_for_all_rows_full: out.n_elem != nrow.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, nrow, "Rosen5d::calc_phi_for_all_rows_full");

    #pragma omp parallel
    {
    arma::vec grad(5, arma::fill::zeros);
    arma::vec hdiag(5, arma::fill::zeros);
    #pragma omp for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      double h12 = 0.0;
      double h14 = 0.0;
      double h23 = 0.0;
      double h45 = 0.0;
      rosen_5d_grad_hess(X(i, 0), X(i, 1), X(i, 2), X(i, 3), X(i, 4),
                         grad, hdiag, h12, h14, h23, h45);
      double trace_term = 0.0;
      for (int d = 0; d < 5; ++d) {
        trace_term += q_cov_c(d, d) * hdiag[d];
      }
      trace_term += 2.0 * (
        q_cov_c(0, 1) * h12 +
        q_cov_c(0, 3) * h14 +
        q_cov_c(1, 2) * h23 +
        q_cov_c(3, 4) * h45
      );
      const double grad_quad = arma::as_scalar(grad.t() * q_cov_c * grad);
      out[i] = 0.5 * (grad_quad + trace_term);
      if (store_components) {
        (*gradient_component)[i] = 0.5 * grad_quad;
        (*hessian_component)[i] = 0.5 * trace_term;
      }
    }
    }
  }

  inline void calc_phi_cube_layer_sums(const arma::cube &particles,
                                 arma::vec &row_sums,
                                 arma::vec &gradient_component,
                                 arma::vec &hessian_component) const {
    // const function as it only changes arma::vec &layer_sums.
    // Takes a N x C x d cube
    // stores the sum of phi across C and d for each particle in layer_sums
    if (particles.n_slices != 5u) {
      Rcpp::stop("Rosen5d phi expects a particle cube with exactly 5 slices.");
    }

    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);

    const arma::mat &X1 = particles.slice(0);
    const arma::mat &X12 = particles.slice(1);
    const arma::mat &X13 = particles.slice(2);
    const arma::mat &X22 = particles.slice(3);
    const arma::mat &X23 = particles.slice(4);

    for (int c = 0; c < nshards; ++c) {
      const double *x1 = X1.colptr(c);
      const double *x12 = X12.colptr(c);
      const double *x13 = X13.colptr(c);
      const double *x22 = X22.colptr(c);
      const double *x23 = X23.colptr(c);
      #pragma omp parallel
      {
      arma::vec grad(5, arma::fill::zeros);
      arma::vec hdiag(5, arma::fill::zeros);
      #pragma omp for schedule(static)
      for (int i = 0; i < npart; ++i) {
        row_sums[i] += phi_rosen_5d(x1[i], x12[i], x13[i], x22[i], x23[i]);
        double h12 = 0.0;
        double h14 = 0.0;
        double h23 = 0.0;
        double h45 = 0.0;
        rosen_5d_grad_hess(x1[i], x12[i], x13[i], x22[i], x23[i],
                           grad, hdiag, h12, h14, h23, h45);
        gradient_component[i] += 0.5 * arma::dot(grad, grad);
        hessian_component[i] += 0.5 * arma::accu(hdiag);
      }
      }
    }
  }

  inline void calc_phi_cube_layer_sums_diagonal(const arma::cube &particles,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    if (particles.n_slices != 5u) {
      Rcpp::stop("Rosen5d phi expects a particle cube with exactly 5 slices.");
    }

    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    if (static_cast<int>(q_diag.n_rows) != nshards || q_diag.n_cols != 5u) {
      Rcpp::stop("Rosen5d diagonal phi received incompatible q_diag dimensions.");
    }
    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);

    for (int c = 0; c < nshards; ++c) {
      #pragma omp parallel
      {
      arma::vec grad(5, arma::fill::zeros);
      arma::vec hdiag(5, arma::fill::zeros);
      #pragma omp for schedule(static)
      for (int i = 0; i < npart; ++i) {
        double h12 = 0.0;
        double h14 = 0.0;
        double h23 = 0.0;
        double h45 = 0.0;
        rosen_5d_grad_hess(particles(i, c, 0), particles(i, c, 1), particles(i, c, 2),
                           particles(i, c, 3), particles(i, c, 4),
                           grad, hdiag, h12, h14, h23, h45);
        double value = 0.0;
        double grad_quad = 0.0;
        double trace_term = 0.0;
        for (int d = 0; d < 5; ++d) {
          const double q_d = q_diag(c, d);
          value += q_d * (grad[d] * grad[d] + hdiag[d]);
          grad_quad += q_d * grad[d] * grad[d];
          trace_term += q_d * hdiag[d];
        }
        row_sums[i] += 0.5 * value;
        gradient_component[i] += 0.5 * grad_quad;
        hessian_component[i] += 0.5 * trace_term;
      }
      }
    }
  }

  inline void calc_phi_cube_layer_sums_full(const arma::cube &particles,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    if (particles.n_slices != 5u) {
      Rcpp::stop("Rosen5d phi expects a particle cube with exactly 5 slices.");
    }

    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    if (static_cast<int>(q_cov.size()) != nshards) {
      Rcpp::stop("Rosen5d full phi expected one q_cov matrix per shard.");
    }
    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);

    for (int c = 0; c < nshards; ++c) {
      const arma::mat &q_cov_c = q_cov[c];
      if (q_cov_c.n_rows != 5u || q_cov_c.n_cols != 5u) {
        Rcpp::stop("Rosen5d full phi received incompatible q_cov dimensions.");
      }
      #pragma omp parallel
      {
      arma::vec grad(5, arma::fill::zeros);
      arma::vec hdiag(5, arma::fill::zeros);
      #pragma omp for schedule(static)
      for (int i = 0; i < npart; ++i) {
        double h12 = 0.0;
        double h14 = 0.0;
        double h23 = 0.0;
        double h45 = 0.0;
        rosen_5d_grad_hess(particles(i, c, 0), particles(i, c, 1), particles(i, c, 2),
                           particles(i, c, 3), particles(i, c, 4),
                           grad, hdiag, h12, h14, h23, h45);
        double trace_term = 0.0;
        for (int d = 0; d < 5; ++d) {
          trace_term += q_cov_c(d, d) * hdiag[d];
        }
        trace_term += 2.0 * (
          q_cov_c(0, 1) * h12 +
          q_cov_c(0, 3) * h14 +
          q_cov_c(1, 2) * h23 +
          q_cov_c(3, 4) * h45
        );
        const double grad_quad = arma::as_scalar(grad.t() * q_cov_c * grad);
        row_sums[i] += 0.5 * (grad_quad + trace_term);
        gradient_component[i] += 0.5 * grad_quad;
        hessian_component[i] += 0.5 * trace_term;
      }
      }
    }
  }

  inline void calc_phi_fused_last_step(const arma::mat &fused,
                                  const int nshards,
                                  arma::vec &row_sums,
                                  arma::vec &gradient_component,
                                  arma::vec &hessian_component) const {
    // Sum of phi over the shards at the fused particles (an N x d matrix).
    // All shards are the same, so \sum_{c=1}^C \phi_c (y) = C \phi_c (y).
    calc_phi_for_all_rows(fused, row_sums, 1, &gradient_component, &hessian_component);
    const double nshards_double = static_cast<double>(nshards);
    row_sums *= nshards_double;
    gradient_component *= nshards_double;
    hessian_component *= nshards_double;
  }

  inline void calc_phi_fused_last_step_diagonal(const arma::mat &fused,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_diag.n_rows);
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows, arma::fill::zeros);
    arma::vec shard_gradient(fused.n_rows, arma::fill::zeros);
    arma::vec shard_hessian(fused.n_rows, arma::fill::zeros);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_diagonal(fused, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_full(const arma::mat &fused,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_cov.size());
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows, arma::fill::zeros);
    arma::vec shard_gradient(fused.n_rows, arma::fill::zeros);
    arma::vec shard_hessian(fused.n_rows, arma::fill::zeros);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_full(fused, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

};


class Logit {
private:
  int nshards_ = 0;
  int ndim_ = 0;
  std::vector<arma::mat> X_; // vector of matrices
  std::vector<arma::vec> Y_;
  std::vector<arma::vec> row_norms_;
  std::vector<double> global_curvature_bound_;


public:
  explicit Logit(const Rcpp::List &params) {
    // constructor: allocate memory and do checks
    if (params.size() < 2) {
      Rcpp::stop("Logit expects params = list(X_list, Y_list).");
    }

    const Rcpp::List X_list = params[0];
    const Rcpp::List Y_list = params[1];
    
    if (X_list.size() != Y_list.size()) {
      Rcpp::stop("Logit expects X_list and Y_list to have the same length (C).");
    }

    nshards_ = static_cast<int>(X_list.size());
    // Reserve the memory, so that push_back below does not reallocate it.
    X_.reserve(nshards_); 
    Y_.reserve(nshards_);
    row_norms_.reserve(nshards_);
    global_curvature_bound_.reserve(nshards_);

    // this bit checks that N and d are the same across all shards.
    int ndim = -1; // i.e. ndim is unset. Slightly ugly, but hey.
    for (int c = 0; c < nshards_; ++c) {
      arma::mat Xc = Rcpp::as<arma::mat>(X_list[c]); // take shard data
      arma::vec Yc = Rcpp::as<arma::vec>(Y_list[c]);

      if (Xc.n_rows != Yc.n_elem) { // check they both have N data points
        Rcpp::stop("Logit expects each shard to satisfy nrow(X) == length(Y).");
      }
      
      if (ndim < 0) { // if ndim is unset then (on the first pass)
        ndim = static_cast<int>(Xc.n_cols); // set it to the number of cols of X
      } else if (static_cast<int>(Xc.n_cols) != ndim) { // o.w. check it's the same
        Rcpp::stop("Logit expects all shards to have the same number of columns.");
      }

      long double max_curvature_sum = 0.0L;
      std::vector<long double> curvature_sums(Xc.n_cols, 0.0L);
      for (arma::uword i = 0; i < Xc.n_rows; ++i) {
        long double row_abs_sum = 0.0L;
        for (arma::uword l = 0; l < Xc.n_cols; ++l) {
          const double x_il = Xc(i, l);
          if (!std::isfinite(x_il)) {
            Rcpp::stop("Logit expects finite design-matrix entries.");
          }
          row_abs_sum += static_cast<long double>(std::abs(x_il));
        }
        for (arma::uword k = 0; k < Xc.n_cols; ++k) {
          curvature_sums[k] += static_cast<long double>(std::abs(Xc(i, k))) * row_abs_sum;
        }
      }
      for (arma::uword k = 0; k < Xc.n_cols; ++k) {
        if (curvature_sums[k] > max_curvature_sum) {
          max_curvature_sum = curvature_sums[k];
        }
      }
      double P_c = static_cast<double>(0.25L * max_curvature_sum);
      if (!std::isfinite(P_c) || P_c < 0.0) {
        Rcpp::stop("Logit global curvature bound is not finite and nonnegative.");
      }
      P_c = std::nextafter(P_c, std::numeric_limits<double>::infinity());
      if (!std::isfinite(P_c) || P_c < 0.0) {
        Rcpp::stop("Logit global curvature bound is not finite and nonnegative after rounding.");
      }

      X_.push_back(std::move(Xc)); // store Xc and Yc and build the complete dataset
      Y_.push_back(std::move(Yc));
      global_curvature_bound_.push_back(P_c);
      
      // precomputes sum of the square norm of each row of X, for each c
      // needed in the calculation of phi.
      // X.back() is Xc, but Xc has just been moved, so this is safer
      row_norms_.push_back(arma::sum(arma::square(X_.back()), 1)); 
    }

    ndim_ = ndim; // finally set the dimension d
  }

  struct LayerBounds {
    double LB;
    double UB;
  };

  inline LayerBounds global_layer_bounds(const Rcpp::List &bessel_layers,
                                         const int shard_1based) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit received an invalid shard index.");
    }
    if (bessel_layers.size() != ndim_) {
      Rcpp::stop("Logit global_layer_bounds expected one layer per parameter dimension.");
    }

    arma::vec theta_hat(ndim_, arma::fill::zeros);
    double D_sq = 0.0;

    for (int r = 0; r < ndim_; ++r) {
      if (TYPEOF(bessel_layers[r]) != VECSXP) {
        Rcpp::stop("Logit global_layer_bounds expected each layer to be a list.");
      }
      const Rcpp::List layer = bessel_layers[r];
      if (!layer.containsElementNamed("L") || !layer.containsElementNamed("U")) {
        Rcpp::stop("Logit global_layer_bounds expected layer elements named L and U.");
      }

      const Rcpp::NumericVector L_vec = layer["L"];
      const Rcpp::NumericVector U_vec = layer["U"];
      if (L_vec.size() != 1 || U_vec.size() != 1) {
        Rcpp::stop("Logit global_layer_bounds expected scalar L and U values.");
      }

      const double L = L_vec[0];
      const double U = U_vec[0];
      if (!std::isfinite(L) || !std::isfinite(U)) {
        Rcpp::stop("Logit global_layer_bounds expected finite L and U values.");
      }
      if (L > U) {
        Rcpp::stop("Logit global_layer_bounds expected L <= U.");
      }

      theta_hat[r] = 0.5 * (L + U);
      const double half_width = 0.5 * (U - L);
      D_sq += half_width * half_width;
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);

    arma::vec linpred = X * theta_hat;
    arma::vec p(ndat);
    for (int i = 0; i < ndat; ++i) {
      p[i] = logistic_stable(linpred[i]);
    }

    const arma::vec grad_hat = X.t() * (Y - p);
    const double D = std::sqrt(D_sq);
    const double P = global_curvature_bound_[shard_index];
    const double grad_norm = arma::norm(grad_hat, 2);

    double LB = -0.5 * static_cast<double>(ndim_) * P;
    double UB = 0.5 * ((grad_norm + D * P) * (grad_norm + D * P) +
                       static_cast<double>(ndim_) * P);

    LB = std::nextafter(LB, -std::numeric_limits<double>::infinity());
    UB = std::nextafter(UB, std::numeric_limits<double>::infinity());

    if (!std::isfinite(LB) || !std::isfinite(UB) || LB > UB) {
      Rcpp::stop("Logit global_layer_bounds produced invalid bounds.");
    }

    return LayerBounds{LB, UB};
  }

  inline void calc_phi_for_all_rows(const arma::mat &Q, //calc_phi_matrix_row_sums
                               arma::vec &out,
                               const int shard_1based,
                               arma::vec *gradient_component = nullptr,
                               arma::vec *hessian_component = nullptr) const {
    // take a Nxd matrix Q of positions
    // write row sums of phi matrix in Nx1 vector out
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit received an invalid shard index.");
    }
    if (static_cast<int>(Q.n_cols) != ndim_) {
      Rcpp::stop("Logit received a matrix with the wrong number of columns.");
    }

    const arma::mat &X = X_[shard_index]; // X_ is a vector of matrices, so X_[shard_index] is a matrix
    const arma::vec &Y = Y_[shard_index];
    const arma::vec &row_norms = row_norms_[shard_index];

    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(Q.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("calc_phi_for_all_rows: (out.n_elem) != npart. ");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "Logit::calc_phi_for_all_rows");

    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec diff(ndat);
    arma::vec grad(ndim_);

    for (int j = 0; j < npart; ++j) {
      linpred = X * Q.row(j).t();
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      diff = Y - p;
      grad = X.t() * diff;
      const double grad_sq = arma::dot(grad, grad);

      double trace_term = 0.0;
      for (int i = 0; i < ndat; ++i) {
        const double pi_i = p[i];
        trace_term -= row_norms[i] * (pi_i * (1.0 - pi_i));
      }

      out[j] = 0.5 * (grad_sq + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_sq;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
  }

  inline void calc_phi_for_all_rows_diagonal(const arma::mat &theta_rows,
                                             arma::vec &out,
                                             const int shard_1based,
                                             const arma::mat &q_diag,
                                             arma::vec *gradient_component = nullptr,
                                             arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("Logit received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_diag.n_rows) != nshards_ || static_cast<int>(q_diag.n_cols) != ndim_) {
      Rcpp::stop("Logit diagonal phi received incompatible q_diag dimensions.");
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(theta_rows.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("calc_phi_for_all_rows_diagonal: out.n_elem != npart.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "Logit::calc_phi_for_all_rows_diagonal");

    const arma::rowvec q_diag_c = q_diag.row(shard_index);
    const arma::vec weighted_row_norms = arma::square(X) * q_diag_c.t();
    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec diff(ndat);
    arma::vec grad(ndim_);

    for (int j = 0; j < npart; ++j) {
      linpred = X * theta_rows.row(j).t();
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      diff = Y - p;
      grad = X.t() * diff;

      double grad_quad = 0.0;
      for (int d = 0; d < ndim_; ++d) {
        grad_quad += q_diag_c[d] * grad[d] * grad[d];
      }

      double trace_term = 0.0;
      for (int i = 0; i < ndat; ++i) {
        trace_term -= weighted_row_norms[i] * (p[i] * (1.0 - p[i]));
      }

      out[j] = 0.5 * (grad_quad + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
  }

  inline void calc_phi_for_all_rows_full(const arma::mat &theta_rows,
                                         arma::vec &out,
                                         const int shard_1based,
                                         const std::vector<arma::mat> &q_cov,
                                         arma::vec *gradient_component = nullptr,
                                         arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("Logit received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_cov.size()) != nshards_) {
      Rcpp::stop("Logit full phi expected one q_cov matrix per shard.");
    }

    const arma::mat &q_cov_c = q_cov[shard_index];
    if (static_cast<int>(q_cov_c.n_rows) != ndim_ || static_cast<int>(q_cov_c.n_cols) != ndim_) {
      Rcpp::stop("Logit full phi received incompatible q_cov dimensions.");
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(theta_rows.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("calc_phi_for_all_rows_full: out.n_elem != npart.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "Logit::calc_phi_for_all_rows_full");

    const arma::vec weighted_row_norms = arma::sum((X * q_cov_c) % X, 1);
    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec diff(ndat);
    arma::vec grad(ndim_);
    arma::vec q_grad(ndim_);

    for (int j = 0; j < npart; ++j) {
      linpred = X * theta_rows.row(j).t();
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      diff = Y - p;
      grad = X.t() * diff;
      q_grad = q_cov_c * grad;
      const double grad_quad = arma::dot(grad, q_grad);

      double trace_term = 0.0;
      for (int i = 0; i < ndat; ++i) {
        trace_term -= weighted_row_norms[i] * (p[i] * (1.0 - p[i]));
      }

      out[j] = 0.5 * (grad_quad + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
  }

  inline void calc_phi_cube_layer_sums(const arma::cube &particles,
                                 arma::vec &row_sums,
                                 arma::vec &gradient_component,
                                 arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of shard columns.");
    }
    if (ndim != ndim_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of parameter dimensions.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows(shard_matrix, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_diagonal(const arma::cube &particles,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of shard columns.");
    }
    if (ndim != ndim_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of parameter dimensions.");
    }
    if (static_cast<int>(q_diag.n_rows) != nshards || static_cast<int>(q_diag.n_cols) != ndim) {
      Rcpp::stop("Logit diagonal phi received incompatible q_diag dimensions.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_diagonal(shard_matrix, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_full(const arma::cube &particles,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of shard columns.");
    }
    if (ndim != ndim_) {
      Rcpp::stop("Logit received a particle cube with the wrong number of parameter dimensions.");
    }
    if (static_cast<int>(q_cov.size()) != nshards) {
      Rcpp::stop("Logit full phi expected one q_cov matrix per shard.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_full(shard_matrix, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step(const arma::mat &fused,
                                  const int nshards,
                                  arma::vec &row_sums,
                                  arma::vec &gradient_component,
                                  arma::vec &hessian_component) const {
    if (nshards != nshards_) {
      Rcpp::stop("Logit received a fused matrix with the wrong number of shards.");
    }

    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards_; ++c) {
      calc_phi_for_all_rows(fused, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_diagonal(const arma::mat &fused,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_diag.n_rows);
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_diagonal(fused, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_full(const arma::mat &fused,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_cov.size());
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_full(fused, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

};


// Logistic regression with a Gaussian prior, where each shard stores the
// distinct (y, x) rows of its data together with their multiplicities. This is
// the full_data_count representation used by Chan et al. (2023). phi depends on
// the data only through sums over rows, so a row occurring count times is
// evaluated once and weighted by count.
class Logit_wprior_countsxypairs {
private:
  int nshards_ = 0;
  int ndim_ = 0;
  std::vector<arma::mat> X_;
  std::vector<arma::vec> Y_;
  std::vector<arma::vec> count_;
  std::vector<arma::vec> row_norms_;
  arma::vec mu0_;
  arma::vec prec0_;

public:
  explicit Logit_wprior_countsxypairs(const Rcpp::List &params) {
    if (params.size() < 5) {
      Rcpp::stop("Logit_wprior_countsxypairs expects params = list(X_list, Y_list, count_list, mu0, prec0).");
    }

    const Rcpp::List X_list = params[0];
    const Rcpp::List Y_list = params[1];
    const Rcpp::List count_list = params[2];

    if (X_list.size() != Y_list.size() || X_list.size() != count_list.size()) {
      Rcpp::stop("Logit_wprior_countsxypairs expects X_list, Y_list and count_list to have the same length (C).");
    }

    nshards_ = static_cast<int>(X_list.size());
    X_.reserve(nshards_);
    Y_.reserve(nshards_);
    count_.reserve(nshards_);
    row_norms_.reserve(nshards_);

    int ndim = -1;
    for (int c = 0; c < nshards_; ++c) {
      arma::mat Xc = Rcpp::as<arma::mat>(X_list[c]);
      arma::vec Yc = Rcpp::as<arma::vec>(Y_list[c]);
      arma::vec count_c = Rcpp::as<arma::vec>(count_list[c]);

      if (Xc.n_rows < 1u) {
        Rcpp::stop("Logit_wprior_countsxypairs expects each shard to have at least one distinct row.");
      }
      if (Xc.n_rows != Yc.n_elem || Xc.n_rows != count_c.n_elem) {
        Rcpp::stop("Logit_wprior_countsxypairs expects each shard to satisfy nrow(X) == length(Y) == length(count).");
      }
      if (ndim < 0) {
        ndim = static_cast<int>(Xc.n_cols);
      } else if (static_cast<int>(Xc.n_cols) != ndim) {
        Rcpp::stop("Logit_wprior_countsxypairs expects all shards to have the same number of columns.");
      }
      if (!Xc.is_finite() || !Yc.is_finite() || !count_c.is_finite()) {
        Rcpp::stop("Logit_wprior_countsxypairs expects finite design, response and count entries.");
      }
      if (count_c.min() < 0.0) {
        Rcpp::stop("Logit_wprior_countsxypairs expects nonnegative counts.");
      }

      X_.push_back(std::move(Xc));
      Y_.push_back(std::move(Yc));
      count_.push_back(std::move(count_c));
      row_norms_.push_back(arma::sum(arma::square(X_.back()), 1));
    }

    ndim_ = ndim;

    mu0_ = Rcpp::as<arma::vec>(params[3]);
    prec0_ = Rcpp::as<arma::vec>(params[4]);
    if (static_cast<int>(mu0_.n_elem) != ndim_ || static_cast<int>(prec0_.n_elem) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs expects mu0 and prec0 to have length equal to the parameter dimension.");
    }
    for (arma::uword i = 0; i < prec0_.n_elem; ++i) {
      if (!std::isfinite(mu0_[i]) || !std::isfinite(prec0_[i]) || prec0_[i] <= 0.0) {
        Rcpp::stop("Logit_wprior_countsxypairs expects finite mu0 and positive finite prec0.");
      }
    }
  }

  inline void calc_phi_for_all_rows(const arma::mat &theta_rows,
                                    arma::vec &out,
                                    const int shard_1based,
                                    arma::vec *gradient_component = nullptr,
                                    arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit_wprior_countsxypairs received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs received a matrix with the wrong number of columns.");
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const arma::vec &count = count_[shard_index];
    const arma::vec &row_norms = row_norms_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(theta_rows.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("Logit_wprior_countsxypairs::calc_phi_for_all_rows: out.n_elem != npart.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart,
        "Logit_wprior_countsxypairs::calc_phi_for_all_rows");

    const double prior_trace = -arma::accu(prec0_);

    #pragma omp parallel
    {
    arma::vec theta(ndim_);
    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec resid(ndat);
    arma::vec grad(ndim_);
    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      theta = theta_rows.row(j).t();
      linpred = X * theta;
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      resid = count % (Y - p);
      grad = X.t() * resid - prec0_ % (theta - mu0_);

      double trace_term = prior_trace;
      for (int i = 0; i < ndat; ++i) {
        trace_term -= count[i] * row_norms[i] * (p[i] * (1.0 - p[i]));
      }

      const double grad_sq = arma::dot(grad, grad);
      out[j] = 0.5 * (grad_sq + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_sq;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_diagonal(const arma::mat &theta_rows,
                                             arma::vec &out,
                                             const int shard_1based,
                                             const arma::mat &q_diag,
                                             arma::vec *gradient_component = nullptr,
                                             arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit_wprior_countsxypairs received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_diag.n_rows) != nshards_ || static_cast<int>(q_diag.n_cols) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs diagonal phi received incompatible q_diag dimensions.");
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const arma::vec &count = count_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(theta_rows.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("Logit_wprior_countsxypairs::calc_phi_for_all_rows_diagonal: out.n_elem != npart.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart,
        "Logit_wprior_countsxypairs::calc_phi_for_all_rows_diagonal");

    const arma::rowvec q_diag_c = q_diag.row(shard_index);
    const arma::vec weighted_row_norms = arma::square(X) * q_diag_c.t();
    double prior_trace = 0.0;
    for (int d = 0; d < ndim_; ++d) {
      prior_trace -= q_diag_c[d] * prec0_[d];
    }

    #pragma omp parallel
    {
    arma::vec theta(ndim_);
    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec resid(ndat);
    arma::vec grad(ndim_);
    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      theta = theta_rows.row(j).t();
      linpred = X * theta;
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      resid = count % (Y - p);
      grad = X.t() * resid - prec0_ % (theta - mu0_);

      double grad_quad = 0.0;
      for (int d = 0; d < ndim_; ++d) {
        grad_quad += q_diag_c[d] * grad[d] * grad[d];
      }

      double trace_term = prior_trace;
      for (int i = 0; i < ndat; ++i) {
        trace_term -= count[i] * weighted_row_norms[i] * (p[i] * (1.0 - p[i]));
      }

      out[j] = 0.5 * (grad_quad + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_full(const arma::mat &theta_rows,
                                         arma::vec &out,
                                         const int shard_1based,
                                         const std::vector<arma::mat> &q_cov,
                                         arma::vec *gradient_component = nullptr,
                                         arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("Logit_wprior_countsxypairs received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_cov.size()) != nshards_) {
      Rcpp::stop("Logit_wprior_countsxypairs full phi expected one q_cov matrix per shard.");
    }

    const arma::mat &q_cov_c = q_cov[shard_index];
    if (static_cast<int>(q_cov_c.n_rows) != ndim_ || static_cast<int>(q_cov_c.n_cols) != ndim_) {
      Rcpp::stop("Logit_wprior_countsxypairs full phi received incompatible q_cov dimensions.");
    }

    const arma::mat &X = X_[shard_index];
    const arma::vec &Y = Y_[shard_index];
    const arma::vec &count = count_[shard_index];
    const int ndat = static_cast<int>(X.n_rows);
    const int npart = static_cast<int>(theta_rows.n_rows);

    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("Logit_wprior_countsxypairs::calc_phi_for_all_rows_full: out.n_elem != npart.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart,
        "Logit_wprior_countsxypairs::calc_phi_for_all_rows_full");

    const arma::vec weighted_row_norms = arma::sum((X * q_cov_c) % X, 1);
    double prior_trace = 0.0;
    for (int d = 0; d < ndim_; ++d) {
      prior_trace -= q_cov_c(d, d) * prec0_[d];
    }

    #pragma omp parallel
    {
    arma::vec theta(ndim_);
    arma::vec linpred(ndat);
    arma::vec p(ndat);
    arma::vec resid(ndat);
    arma::vec grad(ndim_);
    arma::vec q_grad(ndim_);
    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      theta = theta_rows.row(j).t();
      linpred = X * theta;
      for (int i = 0; i < ndat; ++i) {
        p[i] = logistic_stable(linpred[i]);
      }

      resid = count % (Y - p);
      grad = X.t() * resid - prec0_ % (theta - mu0_);
      q_grad = q_cov_c * grad;
      const double grad_quad = arma::dot(grad, q_grad);

      double trace_term = prior_trace;
      for (int i = 0; i < ndat; ++i) {
        trace_term -= count[i] * weighted_row_norms[i] * (p[i] * (1.0 - p[i]));
      }

      out[j] = 0.5 * (grad_quad + trace_term);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_term;
      }
    }
    }
  }

  inline void calc_phi_cube_layer_sums(const arma::cube &particles,
                                       arma::vec &row_sums,
                                       arma::vec &gradient_component,
                                       arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows(shard_matrix, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_diagonal(const arma::cube &particles,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_diagonal(shard_matrix, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_full(const arma::cube &particles,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_full(shard_matrix, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step(const arma::mat &fused,
                                       const int nshards,
                                       arma::vec &row_sums,
                                       arma::vec &gradient_component,
                                       arma::vec &hessian_component) const {
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows(fused, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_diagonal(const arma::mat &fused,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_diag.n_rows);
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_diagonal(fused, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_full(const arma::mat &fused,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_cov.size());
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_full(fused, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }
};


// Weibull proportional hazards model with left truncation, right censoring and
// a random effect for each drive family (Backblaze example).
//  - centred random effects: theta = (alpha, beta, log_rho, log_omega, u_1..u_G)
//    with u_g ~ N(0, omega^2), so eta_g = alpha + beta * s_g + u_g is linear;
//  - omega ~ half-N(0, 0.8^2), written as a density on log_omega;
//  - size_z must be constant within a group, so the likelihood only needs
//    E_g = sum(event), S_g = sum(event * log(stop)) and, per particle, the
//    group sums D_g = sum(stop^rho - start^rho) and its first two derivatives
//    in log(rho). This is exact.
class PHBackblaze_centered_grouped_halfnormal {
private:
  int nshards_ = 0;
  int ngroups_ = 0;
  int ndim_ = 0;
  std::vector<arma::vec> start_;
  std::vector<arma::vec> log_stop_;
  std::vector<arma::vec> log_ratio_;
  std::vector<arma::ivec> group_;
  std::vector<arma::vec> count_;
  std::vector<arma::vec> event_count_;
  std::vector<arma::vec> event_log_stop_;
  arma::vec size_z_group_;
  arma::vec prior_fraction_;


public:
  explicit PHBackblaze_centered_grouped_halfnormal(const Rcpp::List &params) {
    // params = list(start_list, stop_list, event_list, size_z_list, group_list, G, prior_fraction)
    if (params.size() != 7) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects params = list(start_list, stop_list, event_list, size_z_list, group_list, G, prior_fraction).");
    }

    const Rcpp::List start_list = params[0];
    const Rcpp::List stop_list = params[1];
    const Rcpp::List event_list = params[2];
    const Rcpp::List size_z_list = params[3];
    const Rcpp::List group_list = params[4];

    nshards_ = static_cast<int>(start_list.size());
    ngroups_ = Rcpp::as<int>(params[5]);
    prior_fraction_ = Rcpp::as<arma::vec>(params[6]);
    ndim_ = 4 + ngroups_;

    if (nshards_ < 1 || stop_list.size() != nshards_ ||
        event_list.size() != nshards_ || size_z_list.size() != nshards_ ||
        group_list.size() != nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects five nonempty data lists with the same length.");
    }
    if (ngroups_ < 1) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects G to be positive.");
    }
    if (static_cast<int>(prior_fraction_.n_elem) != nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects prior_fraction to have one entry per shard.");
    }
    for (int c = 0; c < nshards_; ++c) {
      if (!std::isfinite(prior_fraction_[c]) ||
          prior_fraction_[c] <= 0.0 || prior_fraction_[c] > 1.0) {
        Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects every prior_fraction entry to be finite and in (0, 1].");
      }
    }

    start_.reserve(nshards_);
    log_stop_.reserve(nshards_);
    log_ratio_.reserve(nshards_);
    group_.reserve(nshards_);
    count_.reserve(nshards_);
    event_count_.reserve(nshards_);
    event_log_stop_.reserve(nshards_);
    size_z_group_.zeros(ngroups_);
    std::vector<bool> size_z_seen(ngroups_, false);

    for (int c = 0; c < nshards_; ++c) {
      arma::vec start_c = Rcpp::as<arma::vec>(start_list[c]);
      arma::vec stop_c = Rcpp::as<arma::vec>(stop_list[c]);
      arma::vec event_c = Rcpp::as<arma::vec>(event_list[c]);
      arma::vec size_z_c = Rcpp::as<arma::vec>(size_z_list[c]);
      arma::ivec group_c = Rcpp::as<arma::ivec>(group_list[c]);
      const arma::uword n = start_c.n_elem;

      if (n < 1u || stop_c.n_elem != n || event_c.n_elem != n ||
          size_z_c.n_elem != n || group_c.n_elem != n) {
        Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects equal, positive data lengths within every shard.");
      }

      arma::vec event_count_c(ngroups_, arma::fill::zeros);
      arma::vec event_log_stop_c(ngroups_, arma::fill::zeros);
      for (arma::uword i = 0; i < n; ++i) {
        if (!std::isfinite(start_c[i]) || !std::isfinite(stop_c[i]) ||
            !std::isfinite(event_c[i]) || !std::isfinite(size_z_c[i]) ||
            start_c[i] < 0.0 || stop_c[i] <= start_c[i] ||
            (event_c[i] != 0.0 && event_c[i] != 1.0) ||
            group_c[i] < 1 || group_c[i] > ngroups_) {
          Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received invalid survival data.");
        }
        const int g = group_c[i] - 1;
        if (!size_z_seen[g]) {
          size_z_group_[g] = size_z_c[i];
          size_z_seen[g] = true;
        } else if (size_z_c[i] != size_z_group_[g]) {
          Rcpp::stop("PHBackblaze_centered_grouped_halfnormal expects size_z to be constant within every group.");
        }
        event_count_c[g] += event_c[i];
        event_log_stop_c[g] += event_c[i] * std::log(stop_c[i]);
      }

      // The event terms are in E_g and S_g, so D_g only depends on the rows
      // through (start, stop, group). Keep one copy of each tied row and carry
      // the multiplicity in count_.
      std::vector<arma::uword> order(n);
      for (arma::uword i = 0; i < n; ++i) {
        order[i] = i;
      }
      const auto tied = [&](arma::uword a, arma::uword b) {
        return start_c[a] == start_c[b] && stop_c[a] == stop_c[b] &&
               group_c[a] == group_c[b];
      };
      std::sort(order.begin(), order.end(), [&](arma::uword a, arma::uword b) {
        if (start_c[a] != start_c[b]) return start_c[a] < start_c[b];
        if (stop_c[a] != stop_c[b]) return stop_c[a] < stop_c[b];
        return group_c[a] < group_c[b];
      });

      arma::vec start_u(n), log_stop_u(n), log_ratio_u(n), count_u(n);
      arma::ivec group_u(n);
      arma::uword m = 0;
      for (arma::uword k = 0; k < n; ++k) {
        const arma::uword i = order[k];
        if (m > 0 && tied(i, order[k - 1])) {
          count_u[m - 1] += 1.0;
          continue;
        }
        start_u[m] = start_c[i];
        log_stop_u[m] = std::log(stop_c[i]);
        log_ratio_u[m] = start_c[i] > 0.0
            ? std::log1p((stop_c[i] - start_c[i]) / start_c[i])
            : 0.0;
        group_u[m] = group_c[i] - 1;
        count_u[m] = 1.0;
        ++m;
      }

      start_.push_back(start_u.head(m));
      log_stop_.push_back(log_stop_u.head(m));
      log_ratio_.push_back(log_ratio_u.head(m));
      group_.push_back(group_u.head(m));
      count_.push_back(count_u.head(m));
      event_count_.push_back(event_count_c);
      event_log_stop_.push_back(event_log_stop_c);
    }
  }

  // Q is the identity, a diagonal, or the full covariance.
  // Every d(eta_g)/d(theta) is the constant vector v_g = e_alpha + s_g e_beta +
  // e_{u_g}, so the likelihood Hessian only needs D_g, B_g = dD_g/dlog(rho) and
  // K_g = d^2 D_g/dlog(rho)^2 in each group.
  inline void calc_gradient_and_hessian_trace(const arma::rowvec &theta,
                                              const int shard_index,
                                              const arma::mat &q_cov,
                                              arma::vec &gradient,
                                              double &hessian_trace) const {
    const arma::vec &start = start_[shard_index];
    const arma::vec &log_stop = log_stop_[shard_index];
    const arma::vec &log_ratio = log_ratio_[shard_index];
    const arma::ivec &group = group_[shard_index];
    const arma::vec &count = count_[shard_index];
    const arma::vec &event_count = event_count_[shard_index];
    const arma::vec &event_log_stop = event_log_stop_[shard_index];
    const double prior_fraction = prior_fraction_[shard_index];
    const double alpha_variance = 4.0;
    const double rho_log_variance = 0.49;
    const double rho_log_mean = std::log(1.2);
    const double omega_variance = 0.64;

    const double alpha = theta[0];
    const double beta = theta[1];
    const double log_rho = theta[2];
    const double log_omega = theta[3];
    const double rho = std::exp(log_rho);
    const double omega_sq = std::exp(2.0 * log_omega);
    const double inv_omega_sq = 1.0 / omega_sq;

    arma::vec D(ngroups_, arma::fill::zeros);
    arma::vec B(ngroups_, arma::fill::zeros);
    arma::vec K(ngroups_, arma::fill::zeros);

    for (arma::uword i = 0; i < start.n_elem; ++i) {
      const double rho_log_stop = rho * log_stop[i];

      // D_i = stop^rho - start^rho. B/D and K/D avoid
      // subtracting nearly equal numbers.
      double D_i = std::exp(rho_log_stop);
      double B_over_D = rho_log_stop;
      double K_over_D = rho_log_stop + rho_log_stop * rho_log_stop;

      if (start[i] > 0.0) {
        const double x = rho * log_ratio[i];
        const double one_minus_exp_minus_x = -std::expm1(-x);
        double q = 0.0;
        if (x == 0.0) {
          q = 1.0;
        } else if (x <= 50.0) {
          q = x / std::expm1(x);
        }

        D_i *= one_minus_exp_minus_x;
        B_over_D = rho_log_stop + q;
        K_over_D = B_over_D + rho_log_stop * rho_log_stop +
                   q * (2.0 * rho_log_stop - x);
      }

      const int g = group[i];
      const double wD = count[i] * D_i;
      D[g] += wD;
      B[g] += wD * B_over_D;
      K[g] += wD * K_over_D;
    }

    gradient.zeros();
    hessian_trace = 0.0;

    for (int g = 0; g < ngroups_; ++g) {
      const int u_index = 4 + g;
      const double s = size_z_group_[g];
      const double eta = alpha + beta * s + theta[u_index];
      const double exp_eta = std::exp(eta);
      const double H = exp_eta * D[g];
      const double HB = exp_eta * B[g];
      const double HK = exp_eta * K[g];
      const double residual = event_count[g] - H;

      gradient[0] += residual;
      gradient[1] += s * residual;
      gradient[2] += event_count[g] + rho * event_log_stop[g] - HB;
      gradient[u_index] += residual;

      const double qv_0 = q_cov(0, 0) + q_cov(0, 1) * s + q_cov(0, u_index);
      const double qv_1 = q_cov(1, 0) + q_cov(1, 1) * s + q_cov(1, u_index);
      const double qv_2 = q_cov(2, 0) + q_cov(2, 1) * s + q_cov(2, u_index);
      const double qv_u = q_cov(u_index, 0) + q_cov(u_index, 1) * s + q_cov(u_index, u_index);
      const double v_q_v = qv_0 + s * qv_1 + qv_u;

      hessian_trace +=
        rho * event_log_stop[g] * q_cov(2, 2) -
        H * v_q_v - 2.0 * HB * qv_2 - HK * q_cov(2, 2);
    }

    // The prior, including u_g ~ N(0, omega^2), is raised to this shard's
    // explicit fractional power. The prior on log_omega includes the Jacobian
    // of omega -> log_omega.
    double sum_u_sq = 0.0;
    for (int g = 0; g < ngroups_; ++g) {
      sum_u_sq += theta[4 + g] * theta[4 + g];
    }
    gradient[0] -= prior_fraction * (alpha + 4.0) / alpha_variance;
    gradient[1] -= prior_fraction * beta;
    gradient[2] -= prior_fraction * (log_rho - rho_log_mean) / rho_log_variance;
    gradient[3] += prior_fraction * (1.0 - omega_sq / omega_variance -
                                     ngroups_ + sum_u_sq * inv_omega_sq);
    for (int g = 0; g < ngroups_; ++g) {
      gradient[4 + g] -= prior_fraction * theta[4 + g] * inv_omega_sq;
    }

    hessian_trace -= prior_fraction *
      (q_cov(0, 0) / alpha_variance +
       q_cov(1, 1) +
       q_cov(2, 2) / rho_log_variance +
       q_cov(3, 3) * 2.0 * (omega_sq / omega_variance + sum_u_sq * inv_omega_sq));
    for (int g = 0; g < ngroups_; ++g) {
      hessian_trace -= prior_fraction * inv_omega_sq *
        (q_cov(4 + g, 4 + g) - 4.0 * q_cov(3, 4 + g) * theta[4 + g]);
    }
  }

  inline void calc_phi_for_all_rows(const arma::mat &Q,
                                    arma::vec &out,
                                    const int shard_1based,
                                    arma::vec *gradient_component = nullptr,
                                    arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received an invalid shard index.");
    }
    if (static_cast<int>(Q.n_cols) != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a matrix with the wrong number of columns.");
    }
    const int npart = static_cast<int>(Q.n_rows);
    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal output length must equal the number of particle rows.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "PHBackblaze_centered_grouped_halfnormal::calc_phi_for_all_rows");

    const arma::mat q_identity(ndim_, ndim_, arma::fill::eye);

    out.zeros(npart);
    #pragma omp parallel
    {
    arma::vec gradient(ndim_);
    double trace_hessian = 0.0;

    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      const arma::rowvec theta = Q.row(j);
      calc_gradient_and_hessian_trace(theta, shard_index, q_identity,
                                      gradient, trace_hessian);
      const double grad_sq = arma::dot(gradient, gradient);
      out[j] = 0.5 * (grad_sq + trace_hessian);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_sq;
        (*hessian_component)[j] = 0.5 * trace_hessian;
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_diagonal(const arma::mat &theta_rows,
                                             arma::vec &out,
                                             const int shard_1based,
                                             const arma::mat &q_diag,
                                             arma::vec *gradient_component = nullptr,
                                             arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_diag.n_rows) != nshards_ || static_cast<int>(q_diag.n_cols) != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal diagonal phi received incompatible q_diag dimensions.");
    }
    const int npart = static_cast<int>(theta_rows.n_rows);
    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal output length must equal the number of particle rows.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "PHBackblaze_centered_grouped_halfnormal::calc_phi_for_all_rows_diagonal");

    const arma::rowvec q_diag_c = q_diag.row(shard_index);
    arma::mat q_diag_dense(ndim_, ndim_, arma::fill::zeros);
    for (int d = 0; d < ndim_; ++d) {
      q_diag_dense(d, d) = q_diag_c[d];
    }

    out.zeros(npart);
    #pragma omp parallel
    {
    arma::vec gradient(ndim_);
    double trace_hessian = 0.0;

    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      const arma::rowvec theta = theta_rows.row(j);
      calc_gradient_and_hessian_trace(theta, shard_index, q_diag_dense,
                                      gradient, trace_hessian);
      double grad_quad = 0.0;
      for (int d = 0; d < ndim_; ++d) {
        grad_quad += q_diag_c[d] * gradient[d] * gradient[d];
      }
      out[j] = 0.5 * (grad_quad + trace_hessian);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_hessian;
      }
    }
    }
  }

  inline void calc_phi_for_all_rows_full(const arma::mat &theta_rows,
                                         arma::vec &out,
                                         const int shard_1based,
                                         const std::vector<arma::mat> &q_cov,
                                         arma::vec *gradient_component = nullptr,
                                         arma::vec *hessian_component = nullptr) const {
    const int shard_index = shard_1based - 1;
    if (shard_index < 0 || shard_index >= nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received an invalid shard index.");
    }
    if (static_cast<int>(theta_rows.n_cols) != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a matrix with the wrong number of columns.");
    }
    if (static_cast<int>(q_cov.size()) != nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal full phi expected one q_cov matrix per shard.");
    }
    const int npart = static_cast<int>(theta_rows.n_rows);
    if (static_cast<int>(out.n_elem) != npart) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal output length must equal the number of particle rows.");
    }
    const bool store_components = prepare_optional_phi_components(
        gradient_component, hessian_component, npart, "PHBackblaze_centered_grouped_halfnormal::calc_phi_for_all_rows_full");

    const arma::mat &q_cov_c = q_cov[shard_index];
    if (static_cast<int>(q_cov_c.n_rows) != ndim_ || static_cast<int>(q_cov_c.n_cols) != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal full phi received incompatible q_cov dimensions.");
    }

    out.zeros(npart);
    #pragma omp parallel
    {
    arma::vec gradient(ndim_);
    arma::vec q_gradient(ndim_);
    double trace_hessian = 0.0;

    #pragma omp for schedule(static)
    for (int j = 0; j < npart; ++j) {
      const arma::rowvec theta = theta_rows.row(j);
      calc_gradient_and_hessian_trace(theta, shard_index, q_cov_c,
                                      gradient, trace_hessian);
      q_gradient = q_cov_c * gradient;
      const double grad_quad = arma::dot(gradient, q_gradient);
      out[j] = 0.5 * (grad_quad + trace_hessian);
      if (store_components) {
        (*gradient_component)[j] = 0.5 * grad_quad;
        (*hessian_component)[j] = 0.5 * trace_hessian;
      }
    }
    }
  }

  inline void calc_phi_cube_layer_sums(const arma::cube &particles,
                                       arma::vec &row_sums,
                                       arma::vec &gradient_component,
                                       arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_ || ndim != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a particle cube with incompatible dimensions.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows(shard_matrix, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_diagonal(const arma::cube &particles,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_ || ndim != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a particle cube with incompatible dimensions.");
    }
    if (static_cast<int>(q_diag.n_rows) != nshards || static_cast<int>(q_diag.n_cols) != ndim) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal diagonal phi received incompatible q_diag dimensions.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_diagonal(shard_matrix, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_cube_layer_sums_full(const arma::cube &particles,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int npart = static_cast<int>(particles.n_rows);
    const int nshards = static_cast<int>(particles.n_cols);
    const int ndim = static_cast<int>(particles.n_slices);

    if (nshards != nshards_ || ndim != ndim_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received a particle cube with incompatible dimensions.");
    }
    if (static_cast<int>(q_cov.size()) != nshards) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal full phi expected one q_cov matrix per shard.");
    }

    row_sums.zeros(npart);
    gradient_component.zeros(npart);
    hessian_component.zeros(npart);
    arma::vec shard_values(npart);
    arma::vec shard_gradient(npart);
    arma::vec shard_hessian(npart);
    arma::mat shard_matrix(npart, ndim);

    for (int c = 0; c < nshards; ++c) {
      for (int d = 0; d < ndim; ++d) {
        shard_matrix.col(d) = particles.slice(d).col(c);
      }
      calc_phi_for_all_rows_full(shard_matrix, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step(const arma::mat &fused,
                                       const int nshards,
                                       arma::vec &row_sums,
                                       arma::vec &gradient_component,
                                       arma::vec &hessian_component) const {
    if (nshards != nshards_) {
      Rcpp::stop("PHBackblaze_centered_grouped_halfnormal received the wrong number of shards.");
    }

    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards_; ++c) {
      calc_phi_for_all_rows(fused, shard_values, c + 1, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_diagonal(const arma::mat &fused,
                                                const arma::mat &q_diag,
                                                arma::vec &row_sums,
                                                arma::vec &gradient_component,
                                                arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_diag.n_rows);
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_diagonal(fused, shard_values, c + 1, q_diag, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }

  inline void calc_phi_fused_last_step_full(const arma::mat &fused,
                                            const std::vector<arma::mat> &q_cov,
                                            arma::vec &row_sums,
                                            arma::vec &gradient_component,
                                            arma::vec &hessian_component) const {
    const int nshards = static_cast<int>(q_cov.size());
    row_sums.zeros(fused.n_rows);
    gradient_component.zeros(fused.n_rows);
    hessian_component.zeros(fused.n_rows);
    arma::vec shard_values(fused.n_rows);
    arma::vec shard_gradient(fused.n_rows);
    arma::vec shard_hessian(fused.n_rows);
    for (int c = 0; c < nshards; ++c) {
      calc_phi_for_all_rows_full(fused, shard_values, c + 1, q_cov, &shard_gradient, &shard_hessian);
      row_sums += shard_values;
      gradient_component += shard_gradient;
      hessian_component += shard_hessian;
    }
  }
};


#endif
