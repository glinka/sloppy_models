#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <dmaps.h>
#include <util_fns.h>
#include "custom_utils.h"
#include "kernel_function.h"

// Performs DMAPS on the the Bryn. lab's data set
// Reads data from params.csv and of_vals.csv, using this as data as input to DMAPS with two different distance measures:
// 1. The standard Euclidean distance between parameter sets
// 2. The extended distance measure motivated by Lafone's thesis that includes obj. fn. information: \f$ d(x_1, x_2) = \frac{\|x_1 - x_2\|^2}{\epsilon} + \frac{| of(x_1) - of(x_2) |}{\epsilon^2} \f$
// /* **Output is saved in ./data/output/dmaps**

typedef std::vector< std::vector<double> > matrix;
typedef std::vector<double> vector;

int main(int argc, char** argv) {
  matrix sloppy_params = util_fns::read_data("./data/input/params.csv");
  std::cout << "finished reading data from ./data/input/params.csv" << std::endl;
  // pull out first and only vector of matrix
  vector of_data = util_fns::read_data("./data/input/of_vals.csv")[0];
  std::cout << "finished reading data from ./data/input/of_vals.csv" << std::endl;
  // scale parameters to have average "1"
  sloppy_params = custom_utils::scale_data(sloppy_params);
  // loop over different epsilon values [1,2,3]
  
    // test test_kernels function over log-spaced epsilons
  const int nkernels = 20;
  const int lower_exp = -4, upper_exp = 4;
  const double de = (upper_exp - lower_exp)/(nkernels - 1.0); // change in epsilon
  vector epsilons(nkernels);
  for (int i = 0; i < nkernels; i++) {
    epsilons[i] = std::pow(10, lower_exp + i*de);
  }
  std::vector<Kernel_Function> kernels;
  for (int i = 0; i < nkernels; i++) {
    kernels.push_back(Kernel_Function(epsilons[i]));
  }
  std::vector<double> kernel_sums = dmaps::test_kernels(sloppy_params, kernels);

  // create file directory and header based on current parameter values
  std::stringstream ss("");
  ss << "npts" << of_data.size();
  std::string file_directory = "data/output/" + ss.str();
  util_fns::create_directory(file_directory);
  std::cout << "saving all data in: " << file_directory << std::endl;
  ss.str("");
  ss << "npts=" << of_data.size();
  std::string file_header = ss.str();

  // save output
  util_fns::save_vector(kernel_sums, file_directory + "/kernel_sums.csv", file_header);
  util_fns::save_vector(epsilons, file_directory + "/epsilons.csv", file_header);
  std::cout << "saved kernel sums as: kernel_sums.csv" << std::endl;
  std::cout << "saved epsilons as: epsilons.csv" << std::endl;

  /* // run dmaps with epsilon = 0.5 for unnormalized ob. fn. vals, 1e-1 for normalized */
  const double epsilon = 1.0;
  Kernel_Function of_kernel(epsilon);
  const int k = 20;
  const double weight_threshold = 1e-8;
  Vector eigvals;
  Matrix eigvects, W;
  dmaps::map(sloppy_params, of_kernel, eigvals, eigvects, W, k, weight_threshold);

  // save DMAPS output: eigvals, eigvects
  // set Eigen ioformat to not align columns and have a comma delimiter between columns
  Eigen::IOFormat comma_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");

  std::ofstream eigvect_file(file_directory + "/eigvects.csv");
  eigvect_file << file_header << std::endl << eigvects.format(comma_format);
  eigvect_file.close();
  std::cout << "saved DMAPS output eigvects as: eigvects.csv" << std::endl;
  std::ofstream eigval_file(file_directory + "/eigvals.csv");
  eigval_file << file_header << std::endl << eigvals.format(comma_format);
  std::cout << "saved DMAPS output eigvals as: eigvals.csv" << std::endl;
  eigval_file.close();

}
