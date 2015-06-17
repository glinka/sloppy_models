#include <random>
#include <functional>
#include "unidentifiable_exp.h"

Unidentifiable_Exp_System::Unidentifiable_Exp_System(const Vector& params, const Vector& times): _params(params), _times(times), _traj(generate_trajectory(params, times)) {
}

double Unidentifiable_Exp_System::objective_function(const Vector& params) const {
  return (generate_trajectory(params, _times) - _traj).squaredNorm();
}

Vector Unidentifiable_Exp_System::generate_trajectory(const Vector& params, const Vector& times) const {
  return (-params(0)*params(1)*_times).array().exp();
}

Matrix Unidentifiable_Exp_System::generate_sloppy_sets(const Vector& params_min, const Vector& params_max, const int nsamples, const double tol) {
  // ? need random seed ?
  std::default_random_engine generator;
  std::uniform_real_distribution<double> uniform_rng(0.0, 1.0);
  auto uniform_rng_wrapper = [&] (const int a) {
    return uniform_rng(generator);
  };
  // concatenate k1 vals and k2 vals into one matrix
  Matrix param_samples = Matrix::Zero(nsamples, 2);
  param_samples.col(0) = params_min(0)*Vector::Ones(nsamples) + (params_max(0) - params_min(0))*param_samples.col(0).unaryExpr(uniform_rng_wrapper);
  param_samples.col(1) = params_min(1)*Vector::Ones(nsamples) + (params_max(1) - params_min(1))*param_samples.col(1).unaryExpr(uniform_rng_wrapper);
  Matrix sloppy_params(nsamples, 3);
  int nsloppy_params = 0;
  // loop through all samples and save if ob. fn. evaluates below 'tol'
  for(int i = 0; i < nsamples; i++) {
    double of_eval = objective_function(param_samples.row(i));
    if(of_eval < tol) {
      sloppy_params(nsloppy_params,0) = param_samples(i,0);
      sloppy_params(nsloppy_params,1) = param_samples(i,1);
      sloppy_params(nsloppy_params,2) = of_eval;
      nsloppy_params++;
    }
  }
  return sloppy_params.topRows(nsloppy_params);
}
