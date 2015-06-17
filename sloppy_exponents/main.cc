#include <sstream>
#include <iostream>
#include <dmaps.h>
#include <util_fns.h>
#include "unidentifiable_exp.h"
#include "kernel_function.h"

int main(int argc, char** argv) {
  // use k1=k2=1
  Vector params(2);
  params << 1, 1;
  double t0 = 0, tf = 1;
  int ntimes = 10;
  Vector times = Vector::LinSpaced(ntimes, t0, tf);
  Unidentifiable_Exp_System system(params, times);
  // sample over box [0.5 -> 1.5]
  Vector params_min(2), params_max(2);
  params_min << 0.98, 0.98;
  params_max << 1.02, 1.02;
  const int nsamples = 10000;
  const double tol = 1e-3;
  Matrix sloppy_params = system.generate_sloppy_sets(params_min, params_max, nsamples, tol);
  std::cout << "found " << sloppy_params.rows() << " sloppy parameter sets" << std::endl;

  // create directory for saved files
  const int npts = sloppy_params.rows();
  std::stringstream ss("");
  ss << "ktrue" << params(0) << "_kmin" << params_min(0) << "_kmax" << params_max(0) << "_npts" << npts << "_tol" << tol;
  std::string file_directory = "data/output/" + ss.str();
  util_fns::create_directory(file_directory);
  std::cout << "saving all data in: " << file_directory << std::endl;

  // create file header for saved output
  ss.str("");
  ss << "ktrue=" << params(0) << ",kmin=" << params_min(0) << ",kmax=" << params_max(0) << ",npts=" << npts << ",tol=" << tol;
  std::string file_header = ss.str();

  // set Eigen ioformat to not align columns and have a comma delimiter between columns
  Eigen::IOFormat comma_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");

  // scale log(obj. fn. values) to have mean one and save
  /* sloppy_params.col(2) = npts*sloppy_params.col(2).array().log()/(sloppy_params.col(2).array().log().sum()); */
  sloppy_params.col(2) = npts*sloppy_params.col(2)/(sloppy_params.col(2).sum());
  std::ofstream param_file(file_directory + "/sloppy_params.csv");
  param_file << file_header << std::endl << sloppy_params.format(comma_format);
  // write everything immediately
  param_file.close();
  std::cout << "saved normalized parameters as: sloppy_params.csv"  << std::endl;
  
  // copy each point into an STL vector entry for DMAPS
  std::vector<Vector> sloppy_params_vec(npts);
  for(int i = 0; i < npts; i++) {
    sloppy_params_vec[i] = sloppy_params.row(i);
  }


  // test test_kernels function over log-spaced epsilons
  const int nkernels = 20;
  const int lower_exp = -8, upper_exp = 1;
  Vector epsilons = Vector::LinSpaced(nkernels, lower_exp, upper_exp);
  for (int i = 0; i < nkernels; i++) {
    epsilons[i] = pow(10, epsilons[i]);
  }
  std::vector<Kernel_Function> kernels;
  for (int i = 0; i < nkernels; i++) {
    kernels.push_back(Kernel_Function(epsilons[i]));
  }
  std::vector<double> kernel_sums = dmaps::test_kernels(sloppy_params_vec, kernels);

  // save output
  util_fns::save_vector(kernel_sums, file_directory + "/kernel_sums.csv", file_header);
  util_fns::save_vector(std::vector<double>(epsilons.data(), epsilons.data()+nkernels), file_directory + "/epsilons.csv", file_header);
  std::cout << "saved kernel sums as: kernel_sums.csv" << std::endl;
  std::cout << "saved epsilons as: epsilons.csv" << std::endl;


  /* // run dmaps with epsilon = 1e-3 for unnormalized ob. fn. vals, 1e-1 for normalized */
  const double epsilon = 1e-1;
  Kernel_Function of_kernel(epsilon);
  const int k = 20;
  const double weight_threshold = 1e-8;
  Vector eigvals;
  Matrix eigvects, W;
  dmaps::map(sloppy_params_vec, of_kernel, eigvals, eigvects, W, k, weight_threshold);

  // save DMAPS output: eigvals, eigvects
  std::ofstream eigvect_file(file_directory + "/eigvects.csv");
  eigvect_file << file_header << std::endl << eigvects.format(comma_format);
  eigvect_file.close();
  std::cout << "saved DMAPS output eigvects as: eigvects.csv" << std::endl;
  std::ofstream eigval_file(file_directory + "/eigvals.csv");
  eigval_file << file_header << std::endl << eigvals.format(comma_format);
  std::cout << "saved DMAPS output eigvals as: eigvals.csv" << std::endl;
  eigval_file.close();

}
