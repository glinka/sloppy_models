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
  std::cout << "finished reading data from ./data/input/params.csv" << std::endl;
  // pull out first and only vector of matrix
  vector of_data = util_fns::read_data("./data/input/of_vals.csv")[0];
  std::cout << "finished reading data from ./data/input/of_vals.csv" << std::endl;
  // scale parameters to have average "1"
  params_data = custom_utils::scale_data(params_data);
  // loop over different epsilon values [1,2,3]
  for(int eps = 1; eps < 4; eps++) {
    // create kernel
    // ? efficiency of choosing epsilon ?
    /* const double eps = 3; // from \sum W_{ij} vs. \epsilon plot */
    Gaussian_Kernel<double> gk(eps);
    vector eigvals; matrix eigvects; matrix W;   // storage for dmaps output
    const int ndims = 20; // number of dimensions to save
    // get embedding
    dmaps::map(params_data, gk, eigvals, eigvects, W, ndims);

    // create directory for this specific value of epsilon if none exists
    std::string id = std::to_string(eps);
    std::string dir = "./data/output/dmaps/euclid/eps_" + id + "/";
    util_fns::create_directory(dir);

    std::cout << "completed DMAP, saving results in " + dir << std::endl;
    // save data using epsilon as file identifier
    // create header for file in form: "metric=euclid,eps=1.0, npts=4000"
    std::string file_header = "metric=euclid,eps=" + std::to_string(eps) + ",npts=" + std::to_string(params_data.size());
    util_fns::save_vector(eigvals, dir + "eigvals.csv", file_header);
    util_fns::save_matrix(eigvects, dir + "eigvects.csv", file_header);
  }
}
