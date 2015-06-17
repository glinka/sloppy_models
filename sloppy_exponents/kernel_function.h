#ifndef KERNEL_FUNCTION_H
#define KERNEL_FUNCTION_H

#include <Eigen/Dense>

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

/* adapation from lafon's thesis which takes into account the objective function value */

class Kernel_Function {
 public:
  // constructor
 Kernel_Function(const double epsilon):_epsilon(epsilon) {}
  // copy constructor
 Kernel_Function(const Kernel_Function& gk):_epsilon(gk._epsilon)  {}
  // move constructor
 Kernel_Function(Kernel_Function&& gk): _epsilon(std::move(gk._epsilon)) {}
  /* no assignment operator, only const members */
  ~Kernel_Function() {}
  double operator()(const Vector& x1, const Vector& x2) const {
    return std::exp(-(x1.head<2>() - x2.head<2>()).squaredNorm()/_epsilon - (x1(2) - x2(2))*(x1(2) - x2(2))/(_epsilon*_epsilon));
  }
 private:
  const double _epsilon;
};

#endif
