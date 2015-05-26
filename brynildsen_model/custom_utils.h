#ifndef _CUSTOM_UTILS_H_
#define _CUSTOM_UTILS_H_

#include <vector>

namespace custom_utils {

  /*
   * Scales each variable to have mean one. Expects input matrix's rows to contain single data points, while columns contain multiple measurements of a single variable.
   * \param data array whose entries will be scaled. data[i] should refer to the \f$ i^{th} \f$ data point.
   * \return scaled data
   */
  std::vector< std::vector<double> > scale_data(const std::vector< std::vector<double> > data);
  /*
   * Checks that each column has been scaled to one by printing the residual between the scale_data output and a ones vector.
   */
  double test_scale_data(const std::vector< std::vector<double> > unscaled_data);

}
  
  

#endif /* _CUSTOM_UTILS_H_ */
