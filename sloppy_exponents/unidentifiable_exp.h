#ifndef _UNIDENTIFIABLE_EXP_H_
#define _UNIDENTIFIABLE_EXP_H_

#include <Eigen/Dense>

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

/**
 * Provides methods for analyzing the objective function derived from the system \f$y=e^{-k_1k_2*t)\f$
 */
class Unidentifiable_Exp_System {
 public:
  /**
   * Assigns true parameter values and sampling times on which to base future evaluations of the least squares objective function, along with generating the corresponding trajectory
   * \param params true parameter values of the system
   * \param times times at which to sample trajectory
   */
  Unidentifiable_Exp_System(const Vector& params, const Vector& times);
  /**
   * Returns the value of \f$C(k_1, k_2) = \sum_i (e^{-\hat{k}_1\hat{k}_2*t_i) - e^{-k_1k_2*t_i))^2\f$, the least squares objective function corresponding to the system \f$y=e^{-k_1k_2*t)\f$
   *
   * \param params parameters (k1, k2) at which to evaluate the ob. fn.
   * \return the objective function evaluated at the given parameter values
   */
  double objective_function(const Vector& params) const;
  /**
   * Returns a data set of \f$\{y_i\}\f$, with \f$y_i = e^{-k_1 k_2 t_i}\f$
   *
   * \param params parameters (k1, k2) with which to generate trajectory
   * \param times 
   */
  Vector generate_trajectory(const Vector& params, const Vector& times) const;
  /**
   * Locates collection of sloppy parameters by sampling over a rectangular grid as specified by params_min and params_max
   *
   * \param params_min lower left-hand corner of rectangle over which to sample: (k1min, k2min)
   * \param params_max upper right-hand corner of rectangle over which to sample: (k1max, k2max)
   * \param nsamples number of samples to draw from rectangle
   * \param tol tolerance for acceptable objective function evaluations. If a \f$C(k_1, k_2) < tol\f$, these parameters are added to the set
   * \return ("number of sloppy parameer sets", 3) sized matrix. Each row stores some \f$\[k_1, k_2, C(k_1, k_2)\]\f$
   */
  Matrix generate_sloppy_sets(const Vector& params_min, const Vector& params_max, const int nsamples, const double tol);
 private:
  const Vector _params; //!< true parameter values around which to calculate the ob. fn.
  const Vector _times; //!< times at which data is drawn
  const Vector _traj; //!< storage for true trajectory created with '_params' sampled at '_times'
};


  

#endif /* _UNIDENTIFIABLE_EXP_H_ */
