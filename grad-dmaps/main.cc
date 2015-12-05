#include <sstream>
#include <iostream>
#include <random>
#include <chrono>
#include <dmaps.h>
#include <util_fns.h>
#include "gradient_kernel.h"

/**
 * Investigates gradient dmaps as a method to uncover level sets of functions
 * For now, investigate \f$f(x,y) = x^2 + y^2\f$
 */



int main(int argc, char** argv) {

  // construct a simple random generator engine from a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  const double amin = 0.1; // min value of rectangle
  const double amax = 1.0; // max value of rectangle
  std::uniform_real_distribution<double> rng(amin, amax);

  // create dataset
  const int npts = 4000;
  std::vector< Vector > pts(npts, Vector(2));
  std::vector< std::vector<double> > pts_copy(npts, std::vector<double>(2));
  for(int i = 0; i < npts; i++) {
    pts[i](0) = rng(generator);
    pts[i](1) = rng(generator);
    pts_copy[i][0] = pts[i](0);
    pts_copy[i][1] = pts[i](1);
  }

  // create file header for saved output
  std::stringstream ss("");
  ss << "amin=" << amin << ",amax=" << amax << ",npts=" << npts;
  std::string file_header = ss.str();

  // set Eigen ioformat to not align columns and have a comma delimiter between columns
  Eigen::IOFormat comma_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");

  util_fns::save_matrix(pts_copy, "./data/dataset.csv", file_header);
  std::cout << "saved dataset in: ./data/dataset.csv"  << std::endl;
  
  // test test_kernels function over log-spaced epsilons
  int nkernels = 20;
  int lower_exp = -8, upper_exp = 1;
  Vector epsilons = Vector::LinSpaced(nkernels, lower_exp, upper_exp);
  for (int i = 0; i < nkernels; i++) {
    epsilons[i] = pow(10, epsilons[i]);
  }
  std::vector<Kernel_Function> kernels;
  for (int i = 0; i < nkernels; i++) {
    kernels.push_back(Kernel_Function(epsilons[i]));
  }
  std::vector<double> kernel_sums = dmaps::test_kernels(pts, kernels);

  // save output
  util_fns::save_vector(kernel_sums, "./data/kernel_sums.csv", file_header);
  util_fns::save_vector(std::vector<double>(epsilons.data(), epsilons.data()+nkernels), "./data/epsilons.csv", file_header);
  std::cout << "saved kernel sums in: ./data/kernel_sums.csv" << std::endl;
  std::cout << "saved epsilons in: ./data/epsilons.csv" << std::endl;

  /* const double epsilon = 1e-1; */
  const int k = 20;
  const double weight_threshold = 1e-8;
  // loop over multiple epsilon values to see how it changes diffusion on levesets
  // create vector of kernels
  nkernels = 5;
  lower_exp = -3, upper_exp = 0;
  epsilons = Vector::LinSpaced(nkernels, lower_exp, upper_exp);
  for(int i = 0; i < nkernels; i++) {
    Kernel_Function grad_kernel(pow(10, epsilons[i]));
    std::cout << "started mapping with epsilon = " << pow(10, epsilons[i]) << "..." << std::flush;
    Vector eigvals;
    Matrix eigvects, W;
    dmaps::map(pts, grad_kernel, eigvals, eigvects, W, k, weight_threshold);
    std::cout << "completed" << std::endl;

    // save DMAPS output: eigvals, eigvects
    std::ofstream eigvect_file("./data/eigvects" + std::to_string(i) +  ".csv");
    eigvect_file << file_header << std::endl << eigvects.format(comma_format);
    eigvect_file.close();
    std::ofstream eigval_file("./data/eigvals" + std::to_string(i) + ".csv");
    eigval_file << file_header << std::endl << eigvals.format(comma_format);
    eigval_file.close();
  }
  std::cout << "saved DMAPS output eigvects as: eigvects.csv" << std::endl;
  std::cout << "saved DMAPS output eigvals as: eigvals.csv" << std::endl;
}
