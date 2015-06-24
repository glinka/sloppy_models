#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <dmaps.h>
#include <util_fns.h>
#include "custom_utils.h"
#include "kernel_function.h"

typedef std::vector< std::vector<double> > matrix;
typedef std::vector<double> vector;

/**
 * Perform DMAP on sloppy parameters from Michaelis Menten model which are generated by mm_sloppiness.py and imported from ./data/input/sloppy_params.csv. Prior to calculating the DMAP, will also investigate the effect of different epsilons in the kernel function.
 */

int main(int argc, char** argv) {
  // import data
  matrix sloppy_params = util_fns::read_data("./data/input/sloppy_params.csv");

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
  vector kernel_sums = dmaps::test_kernels(sloppy_params, kernels);

  // create file directory and header based on current parameter values
  std::stringstream ss("");
  ss << "npts" << sloppy_params.size() << "_nparams" << (sloppy_params[0].size() - 1);
  std::string file_directory = "data/output/" + ss.str();
  util_fns::create_directory(file_directory);
  std::cout << "saving all data in: " << file_directory << std::endl;
  ss.str("");
  ss << "npts=" << sloppy_params.size() << ",nparams=" << (sloppy_params[0].size() - 1);;
  std::string file_header = ss.str();

  // save output
  util_fns::save_vector(kernel_sums, file_directory + "/kernel_sums.csv", file_header);
  util_fns::save_vector(epsilons, file_directory + "/epsilons.csv", file_header);
  std::cout << "saved kernel sums as: kernel_sums.csv" << std::endl;
  std::cout << "saved epsilons as: epsilons.csv" << std::endl;

  /* run dmaps with epsilon = 0.5 for unnormalized ob. fn. vals, 1e-1 for normalized */
  const double epsilon = 0.8;
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


  

  

  
  

  