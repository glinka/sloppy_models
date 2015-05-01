import util_fns as uf
import dmaps
import pca
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

class custom_kernel:
    """A single-function class used to evaluate the modified DMAPS kernel between two points as motivated by Lafone's thesis. That is :math:`W_{ij}=exp(\\frac{\|x_i - x_j\|^2}{\epsilon} - \\frac{(of(x_i) - of(x_j))^2}{\epsilon^2})`

    Attributes:
        _epsilon (float): the DMAPS parameter :math:`\epsilon` to be used in kernel evaluations
    """

    def __init__(self, epsilon):
        # set epsilon
        self._epsilon = epsilon

    def dmaps_of_kernel(self, pt1, pt2):
        """The function used to evaluate :math:`W_{ij}` between 'pt1' and 'pt2' with the prespecified value of :math:`\epsilon`

        Note:
            pt[:-1] contains the parameter vector, while pt[-1] contains the objective function evaluation at that parameter set
        """
        return np.exp(-np.power(np.linalg.norm(pt1[:-1] - pt2[:-1]), 2)/self._epsilon - np.power(pt1[-1] - pt2[-2], 2)/np.power(self._epsilon, 2))
    

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
    # # perform DMAPS over range of possible epsilons as determined from the plot above
    # nepsilons = 6
    # epsilons = np.logspace(4,6,nepsilons)
    # # nepsilons = 1
    # # epsilons = [1e4]
    # ndims = 25
    # eigvals = np.empty((nepsilons, ndims))
    # eigvects = np.empty((nepsilons, n, ndims))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # for eps in enumerate(epsilons):
    #     # eigvals[eps[0]], eigvects[eps[0]] = dmaps.embed_data(params_data, k=ndims, epsilon=eps[1])

    #     eigvals, eigvects = dmaps.embed_data(params_data, k=ndims, epsilon=eps[1])
    #     ax.plot(range(1,ndims+1), eigvals, label=r'$\epsilon=' + str(eps[1]) + '$')
    # ax.set_xlabel('index')
    # ax.set_ylabel('eigenvalue')
    # ax.set_yscale('log')
    # ax.set_xlim((1,ndims))
    # ax.legend()
    # plt.show()

    # # save eigvals for analysis
    # np.savetxt('./brynildsen_model/dmaps_eigvals.csv', eigvals, delimiter=',')
    # # perform DMAPS with customized kernel taken from Lafone's thesis which incorporates obj. fn. info
    # # combine both into one array to pass to DMAPS
    full_data = np.empty((n, p+1))
    full_data[:,:p] = params_data
    full_data[:,p] = of_data
    nepsilons = 6
    epsilons = np.logspace(4,6,nepsilons)
    # nepsilons = 1
    # epsilons = [1e4]
    ndims = 25
    eigvals = np.empty((nepsilons, ndims))
    eigvects = np.empty((nepsilons, n, ndims))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for eps in enumerate(epsilons):
        dmaps_kernel = custom_kernel(eps[1])
        eigvals, eigvects = dmaps.embed_data_customkernel(full_data, ndims, dmaps_kernel.dmaps_of_kernel)
        ax.plot(range(1,ndims+1), eigvals, label=r'$\epsilon=' + str(eps[1]) + '$')
    ax.set_xlabel('index')
    ax.set_ylabel('eigenvalue')
    ax.set_yscale('log')
    ax.set_xlim((1,ndims))
    ax.legend()
    plt.show()
    # print eigvals
    # print 'dmaps eigvals:', eigvals
    # # PCA with correlation matrix
    # pcs, variances = pca.pca(params_data, p, corr=True)
    # print 'singular values:', variances
    # np.savetxt('./brynildsen_model/pca_eigvals.csv', variances, delimiter=',')

if __name__=="__main__":
    main('./brynildsen_model/of_vals.csv', './brynildsen_model/params.csv')
