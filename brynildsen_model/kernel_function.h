#ifndef KERNEL_FUNCTION_H
#define KERNEL_FUNCTION_H
#include <vector>


/* standard gaussian kernel */

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
  double operator()(const std::vector<double>& x1, const std::vector<double>& x2) const {
    double norm = 0;
    for(int i = 0; i < x1.size(); i++) {
      norm += std::pow(x1[i] - x2[i], 2);
    }
    return std::exp(-norm/_epsilon);
  }
 private:
  const double _epsilon;
};

#endif
