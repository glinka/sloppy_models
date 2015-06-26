import argparse
import os
import util_fns as uf
import pca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

import plot_dmaps

def shit_plot():
    optvals = np.genfromtxt('./data/optvals.csv', delimiter=',')
    errs = np.genfromtxt('./data/errs.csv')
    npts = optvals.shape[0]
    npts_toplot = 0
    optvals_toplot = np.empty((npts, 3))
    for i in range(npts):
        if errs[i] < 8e-9:
            optvals_toplot[npts_toplot] = optvals[i]
            npts_toplot = npts_toplot + 1
    print npts_toplot
    optvals_toplot = optvals_toplot[:npts_toplot,:]

    optvals_toplot[:,2] = optvals_toplot[:,2]*1e8
    optvals_toplot[:,2] = optvals_toplot[:,2] - np.amin(optvals_toplot[:,2])
    print optvals_toplot[:,2]
    sing_vals, right_sing_vect = pca.pca(optvals_toplot)
    print sing_vals
    print right_sing_vect
    fig = plt.figure()
    ax = fig.add_subplot(111)
    proj = np.dot(optvals_toplot, right_sing_vect[:,:2])
    ax.scatter(proj[:,0], proj[:,1], c=optvals_toplot[:,2])
    ax.set_ylim((np.amin(proj[:,1]), np.amax(proj[:,1])))
    plt.show(fig)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(optvals_toplot[:,0], optvals_toplot[:,1], optvals_toplot[:npts_toplot,2])
    # ax.set_xlabel('x0')
    # ax.set_ylabel('y0')
    # ax.set_zlabel(r'$\epsilon$')
    # plt.show(fig)

def committee_meeting_sloppiness():
    p = 5
    n = 10
    k1s = np.logspace(-2, 3, n)
    k2s = p/k1s
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k1s, k2s)
    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('./figs/committee/ks.png')
    # add noise as if finding from optimization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = 100
    k = 5
    ps = np.linspace(4, 5, k)
    for p in enumerate(ps):
        k1s = np.logspace(-1, 1, n) + np.random.uniform(size=n)
        k2s = p[1]/k1s
        ax.scatter(k1s, k2s)
    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')
    plt.tight_layout()
    plt.savefig('./figs/committee/noisey_ks.png')
    
def plot_eigvals(eigvals, **kwargs):
    """Plots eigenvalues. Done."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = eigvals.shape[0]
    ax.plot(range(1,n+1), np.sort(eigvals)[::-1], **kwargs)
    ax.scatter(range(1,n+1), np.sort(eigvals)[::-1], zorder=2, lw=2, **kwargs)
    ax.set_xlabel('index')
    ax.set_ylabel('eigenvalue')
    ax.set_xlim((1,n))
    ax.set_yscale('log')
    ax.legend()
    plt.show()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', help="files from which data will be read")
    parser.add_argument('--dmap-embeddings', action='store_true', default=False, help="plot 2d DMAP embeddings from eigenvector inputs")
    parser.add_argument('--kernel-sums', action='store_true', default=False, help="plots kernel sums vs. epsilon in kernel for determination of epsilon in DMAP")
    parser.add_argument('--of-coloring', action='store_true', default=False, help="plots the k1, k2 plane colored by obj. fn. value")
    parser.add_argument('--kplane-coloring', action='store_true', default=False, help="plots k1, k2 plane colored by successive DMAP eigenvectors")
    args = parser.parse_args()
    # import data from files
    # organize each dataset by file header, then by type of data as a dictionary of dictionaries. Each entry of 'dataset' should correspond to dictionary with keys given by 'data_types' and values of actual data. dataset -> data type -> raw data
    # the overarching 'datasets' dict is not meant to be indexed by its keys which are convoluted tuples created from the header, but rather it is intended to be used as an interable in a "for d in datasets" fashion
    datasets = {}
    data_types = ['eigvals', 'eigvects', 'sloppy_params', 'epsilons', 'kernel_sums']
    for filename in args.input_files:
        data, params = uf.get_data(filename, header_rows=1)
        dataset_key = tuple([(key, params[key]) for key in params.keys()])
        if dataset_key not in datasets.keys():
            # no entry currently exists, assign dictionary with entries of empty lists. also assign 'params' entry for dataset dict
            datasets[dataset_key] = {}
            datasets[dataset_key]['params'] = params
        # add data to appropriate dataset, under appropriate 'data_set' key
        for data_type in data_types:
            if data_type in filename:
                datasets[dataset_key][data_type] = data


    # run desired routines over each dataset
    for dataset in datasets.values():
        # plots the k1, k2 plane colored by obj. fn. value
        if args.of_coloring:
            plot_dmaps.plot_xy(dataset['sloppy_params'][:,0], dataset['sloppy_params'][:,1], color=dataset['sloppy_params'][:,2], xlabel=r"$k_1$", ylabel="$k_2$", scatter=True)
        # plots k1, k2 plane colored by successive DMAP eigenvectors
        if args.kplane_coloring:
            # # note the necessity of transposing the eigvects as they are read as row vectors from the file, while the plotting fn. expects column vectors
            # plot_dmaps.plot_embeddings(dataset['eigvects'].T, dataset['eigvals'], dataset['params'])
            # now that we're using Eigen's output, no need to transpose eigvects
            for i in range(1, dataset['eigvects'].shape[1]):
                plot_dmaps.plot_xy(dataset['sloppy_params'][:,0], dataset['sloppy_params'][:,1], color=-dataset['eigvects'][:,i], xlabel=r"$k_1$", ylabel="$k_2$", scatter=True)
        # plots kernel sums vs. epsilon in kernel for determination of epsilon in DMAP
        if args.kernel_sums:
            plot_dmaps.plot_xy(dataset['epsilons'], dataset['kernel_sums'], xlabel=r"$\epsilon$", ylabel="$\sum W_{ij}$", xscale='log', yscale='log')
        if args.dmap_embeddings:
            # color by ob. fn. value if dealing with the k1, k2 sloppy param dataset
            if 'sloppy_params' in dataset.keys():
                # plot_dmaps.plot_embeddings(dataset['eigvects'], np.linspace(1,10,dataset['eigvects'].shape[1]), color=dataset['sloppy_params'][:,2])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(dataset['eigvects'][:,2], dataset['eigvects'][:,5], dataset['eigvects'][:,11])
                ax.set_xlabel('phi3')
                ax.set_ylabel('phi6')
                ax.set_zlabel('phi12')
                plt.show(fig)
                plot_dmaps.plot_embeddings(dataset['eigvects'], dataset['eigvals'], color=dataset['sloppy_params'][:,2], k=12)#plot_3d=True)
            else:
                plot_dmaps.plot_embeddings(dataset['eigvects'], dataset['eigvals'], plot_3d=True)



if __name__=="__main__":
    main()
