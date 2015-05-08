#include "custom_utils.h"

namespace custom_utils {

  std::vector< std::vector<double> > scale_data(const std::vector< std::vector<double> > data) {
    // find dimensions
    const int n = data.size(); // number of data points
    const int m = data[0].size(); // dimensionality of the data, i.e. number of variables to scale
    // create storage for m averages
    std::vector<double> avgs(m, 0);
    // loop over each data point for efficiency, adding values to 'avgs' vector, then rescale each variable at once
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < m; j++) {
	avgs[j] += data[i][j];
      }
    }
    // get actual averages
    for(int i = 0; i < m; i++) {
      avgs[i] /= n;
    }
    // copy data and rescale
    std::vector< std::vector<double> > scaled_data(data);
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < m; j++) {
	scaled_data[i][j] /= avgs[j];
      }
    }
    return scaled_data;
  }

}



  

