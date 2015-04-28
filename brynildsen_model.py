import util_fns as uf
import dmaps
import pca
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

def main(of_filename, params_filename):
    """Computes the DMAPS embedding of a collection of parameter sets that adaquately fit the Brynildsen lab's biological model

    Args:
        of_filename (string): location of csv file containing objective function evaluations of the paramater sets, relative to dirrectory from which this file is run
        params_filename (string): location of csv file containing the paramater sets to be embedded, relative to dirrectory from which this file is run
    """
    # get data
    of_data = uf.get_data(of_filename)
    params_data = uf.get_data(params_filename)
    # dimension of parameter sets
    p = params_data.shape[1]
    # number of data points
    n = params_data.shape[0]
    # # investigate proper epsilon value
    # dmaps.epsilon_plot(np.logspace(-1, 7, 50), params_data, fraction_kept=0.05)
    # embed data
    # ? epsilon val ?
    nepsilons = 6
    epsilons = np.logspace(1,6,nepsilons)
    eigvals = np.empty((nepsilons, p))
    eigvects = np.empty((nepsilons, n))
    for eps in enumerate(epsilons):
        eigvals[eps[0]], eigvects[eps[0]] = dmaps.embed_data(params_data, k=p, epsilon=eps[1])
    np.savetxt('./brynildsen_model/dmaps_eigvals.csv', eigvals, delimiter=',')
    print 'dmaps eigvals:', eigvals
    pcs, variances = pca.pca(params_data, p, corr=True)
    print 'singular values:', variances

if __name__=="__main__":
    main('./brynildsen_model/of_vals.csv', './brynildsen_model/params.csv')
