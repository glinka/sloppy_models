import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gs
import matplotlib.ticker as ticker
import argparse
import os
import util_fns as uf
import pca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
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
    
def plot_of_contours(data, tols=np.logspace(-1,1,20), dir='./figs/contours/'):
    """Plots contours of the MM objective function, where each row of 'data' is (K, V, St, of_eval), looping over unique St values and plotting slices in the K/V plane"""
    tols = np.logspace(2*np.min(data[:-1]), 1, 20)
    # get data
    # data = np.genfromtxt('/home/cbe-ygk-10/holiday/of_evals_tol0.1.csv', delimiter=',')
    # data = np.genfromtxt('./data/of_evals.csv', delimiter=',')
    npts = data.shape[0]
    kmin = np.min(data[:,0]); kmax = np.max(data[:,0])
    vmin = np.min(data[:,1]); vmax = np.max(data[:,1])

    # set up some plotting stuff
    gspec = gs.GridSpec(6,6)
    # set up consistent norm for obj. fn. values
    colornorm = colors.Normalize(vmin=np.log10(np.min(data[:,-1])), vmax=np.log10(np.max(tols)))
    format_fn = lambda val, pos: "%.2f" % val
    # set plot constants
    plt.rcParams['font.size'] = 12

    current_pt = data[0]
    last_index = 0

    # loop over all points, creating separate plot for each value of St
    for k, tol in enumerate(tols):
        for i in range(1,npts):
            # only plot if new St has been found, or if end of file is reached
            if data[i,2] != current_pt[2] or i == npts-1:
                current_pt = data[i]
                data_to_plot = data[last_index:i-1]
                data_to_plot = data_to_plot[data_to_plot[:,-1] < tol]
                last_index = i
                # if there are points, plot them
                if data_to_plot.shape[0] > 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(gspec[:,:5]) # plotting axis
                    ax_cb = fig.add_subplot(gspec[:,5]) # colobar axis
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    plot = ax.scatter(data_to_plot[:,0], data_to_plot[:,1], c=np.log10(data_to_plot[:,3]))
                    ax.set_xlim((kmin, kmax))
                    ax.set_ylim((vmin, vmax))
                    ax.set_xlabel('K')
                    ax.set_ylabel('V')
                    ax.set_title(r'$S_t=$' + str(data[i-1,2]))
                    cb = colorbar.ColorbarBase(ax_cb, cmap='jet', norm=colornorm, orientation='vertical', format=ticker.FuncFormatter(format_fn))
                    plt.savefig(dir + 'contours_S' + str(data[i-1,2]) + '_tol' + str(k) + '.png')
                    plt.close()

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
    parser.add_argument('--param-surface', action='store_true', default=False, help="plots parameters from 'sloppy_params.csv' in three-dimensional space")
    parser.add_argument('--of-contours', action='store_true', default=False, help="plots contours of the objective function, where the input data is a (npts, 4) array whose rows are (K, V, St, of_eval)")
    args = parser.parse_args()
    # import data from files
    # organize each dataset by file header, then by type of data as a dictionary of dictionaries. Each entry of 'dataset' should correspond to dictionary with keys given by 'data_types' and values of actual data. dataset -> data type -> raw data
    # the overarching 'datasets' dict is not meant to be indexed by its keys which are convoluted tuples created from the header, but rather it is intended to be used as an interable in a "for d in datasets" fashion
    datasets = {}
    data_types = ['eigvals', 'eigvects', 'sloppy_params', 'epsilons', 'kernel_sums', 'contours']
    for filename in args.input_files:
        # only import csv files
        if filename[-4:] == ".csv":
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

                # # custom 3d plot, should eventually delete
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                p = ax.scatter(dataset['eigvects'][:,1], dataset['eigvects'][:,2], dataset['eigvects'][:,13], c=dataset['sloppy_params'][:,2])
                ax.set_xlabel(r'$\phi_3$')
                ax.set_ylabel(r'$\phi_6$')
                ax.set_zlabel(r'$\phi_{12}$')
                plt.tick_params(axis='both', which='major', labelsize=0)
                fig.colorbar(p)
                plt.show(fig)
                # end custom plot
                plot_dmaps.plot_embeddings(dataset['eigvects'], dataset['eigvals'], color=dataset['sloppy_params'][:,2], colorbar=True)#plot_3d=True)
            else:
                plot_dmaps.plot_embeddings(dataset['eigvects'], dataset['eigvals'], plot_3d=True)
        if args.param_surface:
            # assume only three parameters have been used for investigation and plot the values in log-space
            plot_dmaps.plot_xyz(dataset['sloppy_params'][:,0], dataset['sloppy_params'][:,1], dataset['sloppy_params'][:,2], xlabel=r'$K_M$', ylabel=r'$V_M$', zlabel=r'$\epsilon$', color=dataset['sloppy_params'][:,2], labelsize=24)
        if args.of_contours:
            print 'loaded data, plotting'
            plot_of_contours(dataset['contours'])


if __name__=="__main__":
    main()
