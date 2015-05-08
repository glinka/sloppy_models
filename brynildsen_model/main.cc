#include <iostream>
#include <fstream>
#include <vector>
#include <dmaps.h>
#include <util_fns.h>
#include "custom_utils.h"
#include "gaussian_kernel.h"

// Performs DMAPS on the the Bryn. lab's data set
// Reads data from params.csv and of_vals.csv, using this as data as input to DMAPS with two different distance measures:
// 1. The standard Euclidean distance between parameter sets
// 2. The extended distance measure motivated by Lafone's thesis that includes obj. fn. information: \f$ d(x_1, x_2) = \frac{\|x_1 - x_2\|^2}{\epsilon} + \frac{| of(x_1) - of(x_2) |}{\epsilon^2} \f$
// /* **Output is saved in ./data/output/dmaps**

typedef std::vector< std::vector<double> > matrix;
typedef std::vector<double> vector;

int main(int argc, char** argv) {
  matrix params_data = util_fns::read_data("./data/input/params.csv");
  // pull out first and only vector of matrix
  vector of_data = util_fns::read_data("./data/input/of_vals.csv")[0];
  // scale parameters to have average "1"
  params_data = custom_utils::scale_data(params_data);
  // create kernel
  // ? efficiency of choosing epsilon ?
  const double eps = 1;
  Gaussian_Kernel<double> gk(eps);
  vector eigvals; matrix eigvects; matrix W;   // storage for dmaps output
  const int ndims = 20; // number of dimensions to save
  // get embedding
  dmaps::map(params_data, gk, eigvals, eigvects, W, ndims);
  // save data
  // create header for file in form: "eps=1.0, npts=4000"
  std::string file_header = "eps=" + std::to_string(eps) + ", npts=" + std::to_string(params_data.size());
  util_fns::save_vector(eigvals, "./data/output/dmaps/euclid/eigvals.csv", file_header);
  util_fns::save_matrix(eigvects, "./data/output/dmaps/euclid/eigvects.csv", file_header);
}
