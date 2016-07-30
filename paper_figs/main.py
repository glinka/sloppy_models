# numpy
import numpy as np
# scipy
from scipy.optimize import minimize
# # PyDSTool
# import PyDSTool as pc
# matplotlib
from matplotlib import colors, colorbar, cm, pyplot as plt, gridspec as gs, tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# mpi
from mpi4py import MPI
# utilities
import os
import sys

# import colormaps
import dmaps
import plot_dmaps
from dmaps_kernels import Data_Kernel
from util_fns import progress_bar
from rawlings_model.main import dmaps_param_set
from rawlings_model.model import Rawlings_Model
from zagaris_model import Z_Model as ZM
from algorithms import Integration
import algorithms.CustomErrors as CustomErrors
from algorithms.PseudoArclengthContinuation import PSA
from algorithms.Derivatives import gradient
import util_fns as uf
from pca import pca

class TwoD_CMAP:
    """Implements a sort of two-dimensional colormap where color = x*c1 + y *c2. Remains to be proven that the colors thus generated are unique on the x/y plane. Meh.


    Attributes:
        xcolor: color along x axis. assumed to start at with white at xmin and transition linearly to xcolor at xmax
        xmin: min value of x data
        xmax: max value of x data

    y variables similarly defined

    """

    def __init__(self, xcolor, xmin, xmax, ycolor, ymin, ymax):
        self._xcolor = xcolor
        self._xmin = xmin
        self._xmax = xmax
        self._xspan = xmax - xmin

        self._ycolor = ycolor
        self._ymin = ymin
        self._ymax = ymax
        self._yspan = ymax - ymin

    def to_rgba(self, xdata, ydata):
        """Converts (n, m) xdata and ydata arrays into a (n, m, 4) rgba color array based on class attributes"""
        rgba_colors = None
        n = xdata.shape[0]
        # if only have length-n vector, return length-n vector of colors
        if len(xdata.shape) is 1:
            rgba_colors = np.empty((n, 4))
            for i in range(n):
                rgba_colors[i] = self._xcolor*(xdata[i] - self._xmin)/self._xspan + self._ycolor*(ydata[i] - self._ymin)/self._yspan
        # else if given grid of x and y points, return grid of colors
        else:
            m = xdata.shape[1]
            rgba_colors = np.empty((n, m, 4))
            for i in range(n):
                for j in range(m):
                    rgba_colors[i,j] = self._xcolor*(xdata[i,j] - self._xmin)/self._xspan + self._ycolor*(ydata[i,j] - self._ymin)/self._yspan
        return rgba_colors/np.max(rgba_colors, axis=(0,1))
        


def rgb_square_cmap(xvals, yvals):
    """attempt to figure out Illustrator's 'mix' option"""
    blue = np.array((0,0,1,0.5))
    red = np.array((1,0,0,0.5))
    npts = 500
    xgrid, ygrid = np.meshgrid(np.linspace(0,1,npts), np.linspace(0,1,npts))
    cs = np.empty((npts,npts,4))
    for i in range(npts):
        for j in range(npts):
            cs[i,j] = xgrid[i,j]*blue + ygrid[i,j]*red
            cs[i,j,3] = np.power((xgrid[i,j] + ygrid[i,j])/2.0, 0.5) #(np.power(xgrid[i,j], 2) + np.power(ygrid[i,j], 2))/2.0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cs)
    plt.show()



def rawlings_keff_delta_figs():
    """Produces set of images that attempt to clearly illustrate how increasing \delta(\theta) radius leads to wider range of k_{eff}, and also that the direction of changing \delta(\theta) is exactly the direction of changing k_{eff}"""
    data = np.load('../rawlings_model/data/params-ofevals.pkl')

    gsize = 40
    gspec = gs.GridSpec(gsize,gsize)
    goffset = 5 # due to the large amount of unecessary whitespace included in the 3d plots, we must shrink the axes of the line plots to align nicely: this offset serves that purpose
    fontsize = 72

    # plot points that fall some delta for three distinct delta values
    logdelta_min = -5
    logdelta_max = -3
    ndeltas = 3
    deltas = np.logspace(logdelta_min, logdelta_max, ndeltas)

    npts_to_plot = 3000
    scatter_alpha = 0.07
    logdelta_line_min = -6 # min delta for 1d line plot showing delta range
    nline_pts = 100

    # constants for eventual surface plots
    k1min, k1max = np.min(data[:,1]), np.max(data[:,1])
    kinvmin, kinvmax = np.min(data[:,2]), np.max(data[:,2])
    k2min, k2max = np.min(data[:,3]), np.max(data[:,3])
    npts = 200
    stride = 2
    kinvs, k2s = np.meshgrid(np.logspace(np.log10(kinvmin), np.log10(kinvmax), npts), np.logspace(np.log10(k2min), np.log10(k2max), npts))
    surface_alpha = 0.5

    xlims = None; ylims = None; zlims = None;

    # calculate max range for keff for keff line plot
    temp_data = data[data[:,0] < deltas[-1]]
    temp_keffs = temp_data[:,1]*temp_data[:,3]/(temp_data[:,2] + temp_data[:,3])
    keffmin, keffmax = np.min(temp_keffs), np.max(temp_keffs)
    keff_ticks = np.linspace(keffmin, keffmax, 5)

    # # set various linewidth constants
    lw_adjustment_factor = 1.002 # if you use 1.0 as would normally be expected, matplotlib's line exceeds its actual limits and it appears as though a slightly different range of data is being plotted (-5 to -3.9 instead of -5 to -4), scaling the larger number by this small amount adjusts for this discrepancy
    data_lw = 20 # lw of actual plotted line
    xaxis_lw = 3 # lw of xaxis
    major_tick_lw = 3
    minor_tick_lw = 2
    major_tick_length = 20
    minor_tick_length = 10

    red_value = 204./255 # r-value for surfaces

    figs = []
    axs = []
    axlines = []    
        
    for i, delta in enumerate(deltas):
        # # # plot data
        fig1 = plt.figure()
        figs.append(fig1)

        # # plot data points in 3d
        ax1 = fig1.add_subplot(gspec[:gsize-3,:], projection='3d')
        axs.append(ax1)
        
        logdata = np.log10(data[data[:,0] < delta])
        slice = np.max((logdata.shape[0]/npts_to_plot, 1)) # slice should always be at least 1
        ax1.scatter(logdata[:,1], logdata[:,2], logdata[:,3], alpha=scatter_alpha)


        # # line plot showing log(delta) range
        axline1 = fig1.add_subplot(gspec[gsize-3:,goffset:gsize-goffset])
        axlines.append(axline1)

        axline1.set_xscale('log')

        # plot data on xaxis (y=0), so have yaxis range be (-0.5, 0.5)
        delta_data = np.logspace(lw_adjustment_factor*np.log10(delta), logdelta_line_min, nline_pts)
        axline1.plot(delta_data, 0.0*np.ones(nline_pts), lw=data_lw)

        # do some formatting and labelling
        axline1.set_xlim(np.power(10, (logdelta_line_min, logdelta_max)))
        axline1.set_xlabel(r'$\log(\delta(\theta))$')
        axline1.text(np.power(10, logdelta_line_min - 0.2), -0.06, r'$\ldots$', fontsize=72) # pretty fragile text placement as usual
        axline1.tick_params(labelsize=72)
        


        # first time through, grab lims to keep consistent through all plots
        if xlims is None:
            xlims = ax1.get_xlim()
            ylims = ax1.get_ylim()
            zlims = ax1.get_zlim()


        # # # plot keff surface
        fig2 = plt.figure()
        figs.append(fig2)

        # # plot data points in 3d
        ax2 = fig2.add_subplot(gspec[:gsize-3,:], projection='3d')
        axs.append(ax2)

        # calculate k1, k2, kinv values on surf and plot
        trimmed_data = data[data[:,0] < delta]
        keffs = trimmed_data[:,1]*trimmed_data[:,3]/(trimmed_data[:,2] + trimmed_data[:,3])
        keff_surfs = [np.min(keffs), np.max(keffs)]
        for keff in keff_surfs:
            k1s = keff*(kinvs + k2s)/k2s
            cs = np.ones((npts,npts,4))*np.array((red_value,0,0,surface_alpha)) # change color and alpha
            for j in range(npts):
                for k in range(npts):
                    if k1s[j,k] > 0.99*k1max:
                        cs[j,k,3] = 0
            ax2.plot_surface(np.log10(k1s), np.log10(kinvs), np.log10(k2s), linewidth=0, facecolors=cs, edgecolors=np.array((0,0,0,0)), shade=False, cstride=stride, rstride=stride, antialiased=True)

        # # line plot showing keff range
        axline2 = fig2.add_subplot(gspec[gsize-3:,goffset:gsize-goffset])
        axlines.append(axline2)

        keff_data = np.linspace(np.min(keffs), np.max(keffs), nline_pts)

        # plot data on xaxis (y=0), so have yaxis range be (-0.5, 0.5)
        axline2.plot(keff_data, 0.0*np.ones(nline_pts), lw=data_lw, color=(red_value,0,0,1))

        # set xticks
        axline2.set_xticks(keff_ticks) # silly matplotlib insists on adding two additional intermediate ticks
        axline2.set_xticklabels(['%1.3f' % k for k in keff_ticks])
        axline2.tick_params(labelsize=72)

        # fatten and lengthen xaxis ticks
        axline2.set_xlabel(r'$k_{eff}$')
        axline2.set_xlim((keffmin, keffmax))

        # # # if on largest delta, plot both data and keff on same plot WITH LABELS
        if i == 100: #ndeltas-1:
            # don't format fig3 with other figures
            fig3 = plt.figure()
            fig3.subplots_adjust(bottom=0.11, left=0.07, right=0.93)

            # don't format ax3 with the other 3d axes
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.scatter(logdata[:,1], logdata[:,2], logdata[:,3], alpha=scatter_alpha)
            for keff in keff_surfs:
                k1s = keff*(kinvs + k2s)/k2s
                cs = np.ones((npts,npts,4))*np.array((red_value,0,0,surface_alpha)) # change color and alpha
                for i in range(npts):
                    for j in range(npts):
                        if k1s[i,j] > 0.99*k1max:
                            cs[i,j,3] = 0
                ax3.plot_surface(np.log10(k1s), np.log10(kinvs), np.log10(k2s), linewidth=0, facecolors=cs, edgecolors=np.array((0,0,0,0)), shade=False, cstride=stride, rstride=stride, antialiased=True)

            # format 3d axis
            ax3.view_init(elev=240, azim=70)
            ax3.set_xlim(xlims)
            # hardcoded axes ticks!
            ax3.set_xticks(np.linspace(-1, -1.2, 3))
            ax3.set_xlabel('\n\n' + r'$\log(k_1)$')
            ax3.set_ylim(ylims)
            ax3.set_yticks(np.linspace(1, 4, 3))
            ax3.set_zlim(zlims)
            ax3.set_ylabel('\n\n' + r'$\log(k_{-1})$') 
            ax3.set_zticks(np.linspace(1, 4, 3))
            ax3.set_zlabel('\n\n' + r'$\log(k_2)$')


            fig4 = plt.figure()
            axline3_delta = fig4.add_subplot(111)
            axlines.append(axline3_delta)
            axline3_delta.set_xscale('log')

            axline3_delta.plot(delta_data, 0.0*np.ones(nline_pts), lw=data_lw)
            # do some formatting and labelling
            axline3_delta.set_xlim(np.power(10, (logdelta_line_min, logdelta_max)))
            axline3_delta.set_xlabel(r'$\log(\delta(\theta))$')
            axline3_delta.text(np.power(10, logdelta_line_min - 0.12), 0.0, r'$\ldots$') # pretty fragile text placement as

            fig5 = plt.figure()
            axline3_keff = fig5.add_subplot(111)
            axlines.append(axline3_keff)

            # plot data on xaxis (y=0), so have yaxis range be (-0.5, 0.5)
            axline3_keff.plot(keff_data, 0.0*np.ones(nline_pts), lw=data_lw, color=(red_value,0,0,1))

            # set xticks
            axline3_keff.set_xticks(keff_ticks) # silly matplotlib insists on adding two additional intermediate ticks
            axline3_keff.set_xticklabels(['%1.3f' % k for k in keff_ticks])

            # fatten and lengthen xaxis ticks
            axline3_keff.set_xlabel(r'$k_{eff}$')
            axline3_keff.set_xlim((keffmin, keffmax))
            

        # # # format figures
        for ax in axs:
            # show grid
            ax.grid(True)
            # hide labels
            ax.set_xticklabels([]) 
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            # hide tick labels
            ax.tick_params(color=(0,0,0,0))
            # set 3d viewing angle
            ax.view_init(elev=240, azim=70)
            # set consistent axis limits
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)

        for ax in axlines:
            # set yaxis limits and hide actual axis
            ax.set_ylim((-0.5, 0.5))
            ax.yaxis.set_visible(False)

            # hide left, right and top spines, only adjust bottom by moving to middle of plot and fattening
            ax.spines['bottom'].set_position('zero')
            ax.spines['bottom'].set_color('k')
            ax.spines['bottom'].set_linewidth(xaxis_lw)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.xaxis.set_ticks_position('bottom') # not sure why this is necessary but it is

            # hide white background
            ax.patch.set_alpha(0) 

            # fatten and lengthen xaxis ticks
            ax.tick_params(axis='x', which='major', pad=25, width=major_tick_lw, length=major_tick_length, direction='inout')
            ax.tick_params(axis='x', which='minor', pad=25, width=minor_tick_lw, length=minor_tick_length, direction='inout')

        for fig in figs:
            # center plot left/right, push it up to see bottom label
            fig.subplots_adjust(bottom=0.24, left=0.07, right=0.93, top=1.0)

    plt.show()



def linear_sing_pert_param_space_log_fig():
    """plots output from auto goodly"""
    # load pre-computed data
    lcsplus_large = np.load('c-code/linearlc-fullplus-largedelta.pkl')
    lcsminus_large = np.load('c-code/linearlc-fullminus-largedelta.pkl')
    lcsplus_small = np.load('c-code/linearlc-fullplus-smalldelta.pkl')
    lcsminus_small = np.load('c-code/linearlc-fullminus-smalldelta.pkl')

    # # combine datasets into just two arrays for plus and minus curves
    # plus branches
    npl = lcsplus_large.shape[0]
    nps = lcsplus_small.shape[0]
    lcsplus = np.empty((nps + npl, lcsplus_large.shape[1]))
    lcsplus[:nps] = lcsplus_small
    lcsplus[nps:] = lcsplus_large
    # minus branches
    nml = lcsminus_large.shape[0]
    nms = lcsminus_small.shape[0]
    lcsminus = np.empty((nms + nml, lcsminus_large.shape[1]))
    lcsminus[:nms] = lcsminus_small
    lcsminus[nms:] = lcsminus_large

    lcsplus[:,3] = np.log10(lcsplus[:,3])
    lcsminus[:,3] = np.log10(lcsminus[:,3])

    # loop through each unique value of lcs[:,3], the delta value

    branches = []
    current_delta = lcsplus[0,3]
    start = 0
    npts = lcsplus.shape[0]
    for i in range(npts):
        if lcsplus[i,3] != current_delta:
            branches.append(lcsplus[start:i])
            current_delta = lcsplus[i,3]
            start = i
    branches.append(lcsplus[start:npts])
    nbranches = len(branches)

    print 'have', nbranches, 'branches'
        
    # # # remove any branches for which 1/eps goes below zero
    # kept_pts = np.empty((npts, 4))
    # nkept_branches = 0
    # for i in range(nbranches):
    #     branch = lcs[i*npts_per_branch:(i+1)*npts_per_branch]
    #     if branch[branch < 0].shape[0] == 0:
    #         kept_pts[nkept_branches*npts_per_branch:(nkept_branches+1)*npts_per_branch] = np.copy(lcs[i*npts_per_branch:(i+1)*npts_per_branch])
    #         nkept_branches = nkept_branches + 1
    # # relabel kept pts
    # nbranches = nkept_branches
    # lcs = kept_pts
    # npts = lcs.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # axins = inset_axes(ax, 11, 6, bbox_to_anchor=(0.7, 0.95), bbox_transform=ax.transAxes)

    # # plot each branch
    # first the 'plus' branches
    lw = 3
    lw_inset = 1
    bslice = 50
    custom_indices = [0, 10, 20, 40] # [0, 18, 100, 200, 209] # for smalldelta data

    # set up colormap
    colornorm = colors.Normalize(branches[custom_indices[0]][0,3], branches[custom_indices[-1]][0,3])
    colormap = cm.ScalarMappable(norm=colornorm)

    vT = None
    scale = None
    center = None

    print branches[0].shape
    for i in custom_indices:
        branch = branches[i] 
        branch[:,1] = np.log(branch[:,1])

        if i == custom_indices[0]:
            center = np.average(branch[:,:2], axis=0)
            u, s, vT = np.linalg.svd(branch[:,:2] - center, full_matrices=False)
            scale = s[1]

        vcomps = np.dot(branch[:,:2] - center, vT.T)
        vcomps[:,0] = scale*vcomps[:,0]
        branch[:,:2] = np.dot(vcomps, vT) + center
        ax.plot(branch[:,0], branch[:,1], color=colormap.to_rgba(branch[0,3]), lw=lw)

    # # # now do some svd and stretch the shortest axis of the the smallest two branches and plot these in the inset so that their actual ellipsoidal nature is clear
    # branch0 = branches[0][:,:2]
    # col_avgs = np.average(branch0, axis=0)
    # u, s, v = np.linalg.svd(branch0 - col_avgs, full_matrices=False)
    # scale = 10

    # # now scale the other branches in the same direction
    # for i in range(1):
    #     branch = branches[custom_indices[i]][:,:2]
    #     # col_avgs = np.average(branch, axis=0)
    #     y = np.dot(branch - col_avgs, np.transpose(v))
    #     y[:,1] = scale*y[:,1]
    #     branch = np.dot(y, v) + col_avgs
    #     # axins.plot(branch[:,0], np.log(branch[:,1]), color=colormap.to_rgba(branches[custom_indices[i]][0,3]))


    # second, the 'minus' branches
    branches = []
    current_delta = lcsminus[0,3]
    npts = lcsminus.shape[0]
    start = 0
    for i in range(npts):
        if lcsminus[i,3] != current_delta:
            branches.append(lcsminus[start:i])
            current_delta = lcsminus[i,3]
            start = i
    branches.append(lcsminus[start:npts])

    for i in custom_indices:
        branch = branches[i] 
        branch[:,1] = np.log(branch[:,1])

        vcomps = np.dot(branch[:,:2] - center, vT.T)
        vcomps[:,0] = scale*vcomps[:,0]
        branch[:,:2] = np.dot(vcomps, vT) + center
        ax.plot(branch[:,0], branch[:,1], color=colormap.to_rgba(branch[0,3]), lw=lw)

    # # now scale the other branches in the same direction
    # for i in range(1):
    #     branch = branches[custom_indices[i]][:,:2]
    #     # col_avgs = np.average(branch, axis=0)
    #     y = np.dot(branch - col_avgs, np.transpose(v))
    #     y[:,1] = scale*y[:,1]
    #     branch = np.dot(y, v) + col_avgs
    #     # axins.plot(branch[:,0], np.log(branch[:,1]), color=colormap.to_rgba(branches[custom_indices[i]][0,3]))



    # # tidy up
    ax.set_xlabel(r'$1/\epsilon^*$')
    ax.set_ylabel(r'$\log(x_0)^*$')
    ax.set_ylim((-0.03,0.06))
    ax.set_xlim((4.97,5.07))
    plt.locator_params(nbins=6)
    fig.subplots_adjust(bottom=0.15, left=0.15)

    # # add an inset marker
    # # axins.set_xlim((4.9, 5.1))
    # # axins.set_ylim((0.9, 1.1))
    # # axins.tick_params(axis='both', which='major', labelsize=0)
    # # mark_inset(ax, axins, loc1=4, loc2=3, fc='none', ec='0.5', zorder=3)

    plt.show()


def henon(x0, y0, n, a, b):
    if n > 0:
        return henon(1 - a*x0*x0 + y0, b*x0, n-1, a, b)
    else:
        return [x0, y0]

def henon_inv(xn, yn, n, a, b):
    if n > 0:
        return henon_inv(yn/b, xn - 1 + a*np.power(yn/b, 2), n-1, a, b)
    else:
        return [xn, yn]

def henon_3inv(x3, y3, a, b):
    y0 = (x3 - 1 + a*np.power(y3/b, 2))/b - 1 + a*np.power((y3/b - 1 + a*np.power((x3 - 1 + a*np.power(y3/b, 2))/b, 2))/b, 2)
    x0 = (y3/b - 1 + a*np.power((x3 - 1 + a*np.power(y3/b, 2))/b, 2))/b
    return x0, y0

def henon_2inv(x2, y2, a, b):
    y0 = y2/b - 1 + a*np.power((x2 - 1 + a*np.power(y2/b, 2))/b, 2)
    x0 = (x2 - 1 + a*np.power(y2/b, 2))/b
    return x0, y0

class Transform_Zagaris_Model:
    """Uses Henon map to transform params (alpha, lambda), keeps (beta, epsilon) constant, provides coresponding of"""
    def __init__(self, base_params, times, x0, n, a, b):
        self._beta = base_params[1]
        self._epsilon = base_params[3]
        self._system = ZM.Z_Model(base_params)
        self._true_trajectory = self._system.get_trajectory_quadratic(x0, times)
        self._n = n
        self._a = a
        self._b = b
        self._times = times
        self._x0 = x0

    def of(self, params):
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam, alpha = henon_inv(params[0], params[1], self._n, self._a, self._b)
        self._system.change_parameters(np.array((alpha, self._beta, lam, self._epsilon)))
        try:
            new_traj = self._system.get_trajectory_quadratic(self._x0, self._times)
        except CustomErrors.IntegrationError:
            raise
        else:
            return np.linalg.norm(new_traj - self._true_trajectory)
    
    def henon_2inv_dmaps_metric(self, transformed_params1, transformed_params2):
        """Inverts transformed params twice to uncover alpha, lambda, then take distance there"""
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam_alpha1 = np.array(henon_inv(transformed_params1[0], transformed_params1[1], self._n, self._a, self._b))
        lam_alpha2 = np.array(henon_inv(transformed_params2[0], transformed_params2[1], self._n, self._a, self._b))
        return np.linalg.norm(lam_alpha2 - lam_alpha1)

class Normal_Zagaris_Model:
    """Uses Henon map to transform params (alpha, lambda), keeps (beta, epsilon) constant, provides coresponding of"""
    def __init__(self, base_params, times, x0):
        self._beta = base_params[1]
        self._epsilon = base_params[3]
        self._system = ZM.Z_Model(base_params)
        self._true_trajectory = self._system.get_trajectory_quadratic(x0, times)
        self._times = times
        self._x0 = x0

    def of(self, params):
        # NOTE THAT LAMBDA IS RETURNED FIRST, THEN ALPHA
        lam, alpha = params
        self._system.change_parameters(np.array((alpha, self._beta, lam, self._epsilon)))
        try:
            new_traj = self._system.get_trajectory_quadratic(self._x0, self._times)
        except (CustomErrors.IntegrationError, TypeError):
            raise
        else:
            return np.linalg.norm(new_traj - self._true_trajectory)


def transformed_param_space_fig():
    """Perform DMAP on nonlinear, henon-mapped parameters lambda/epsilon (important/sloppy)"""
    
    # CREATE DATASET (no dataset exists):
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # set model params
    (a_true, b_true, lam_true, eps_true) = (1.0, 0.01, 1.0, 0.001) # (0.1, 0.01, 0.1, 0.001)
    params = np.array((a_true, b_true, lam_true, eps_true))

    # set up integration times
    t0 = 0
    tf = 1
    dt = 0.1
    times = np.arange(0, tf, dt) + dt
    ntimes = times.shape[0]

    # get true trajectory based on true initial conditions
    x0 = np.array((1, a_true))

    # set henon transform params
    nhenon_transforms = 2
    a = 1.4
    b = 0.3

    # create system with given params
    z_system = Transform_Zagaris_Model(params, times, x0, nhenon_transforms, a, b)

    # assume if not running on multiple procs, then we already have data. otherwise re-generate
    if nprocs > 1:

        # perform optimization in transformed parameter space
        do_transformed_optimization = True
        if do_transformed_optimization:
            if rank == 0:
                print 'running in parallel, generating transformed x2, y2 parameters through repeated optimization'
            # optimize over range of TRANSFORMED initial conditions
            nsamples = 1000000
            data = np.empty((nsamples, 3))
            # x2_y2_samples = np.random.uniform(size=(nsamples, 2))*np.array((1, 0.2)) + np.array((1, -0.2))
            x2_y2_samples = np.random.uniform(size=(nsamples, 2))*np.array((32,2.2)) + np.array((-30, -1.5)) # a \in (7, 9) lamb \in (6, 11)
            count = 0
            tol = 1.0
            for params in uf.parallelize_iterable(x2_y2_samples, rank, nprocs):
                try:
                    of_val = z_system.of(params)
                    pt = np.empty(3)
                    if of_val < tol:
                        pt = (params[0], params[1], of_val)
                    else:
                        result = minimize(z_system.of, params, method='SLSQP')#, tol=tol, options={'ftol' : tol})
                        pt = (result.x[0], result.x[1], result.fun)
                except (CustomErrors.IntegrationError, TypeError):
                    continue
                else:
                    # if result.success:
                    data[count] = pt
                    count = count + 1

            data = data[:count]
            all_data = comm.gather(data, root=0)

            if rank is 0:
                all_data = np.concatenate(all_data)
                all_data.dump('./data/x2-y2-ofevals-2016.csv')
                print '******************************'
                print 'Data saved in ./data/x2-y2-ofevals-2016.csv, rerun to perform DMAP'
                print '******************************'


        # perform optimization in normal parameter space
        do_normal_optimization = False
        if do_normal_optimization:
            if rank == 0:
                print 'running in parallel, generating normal alpha, lambda parameters through repeated optimization'

            z_system = Normal_Zagaris_Model(params, times, x0)

            nsamples = 10000
            data = np.empty((nsamples, 3))
            data2 = np.empty((nsamples, 3))
            scale = 3
            a_lam_samples = np.random.uniform(size=(nsamples,2))*np.array((scale, scale)) # a \in [0,scale], lam \in [0,scale]
            # offset = 1
            # a_lam_samples = np.array((a_true - offset/2.0, lam_true - offset/2.0)) + np.random.uniform(size=(nsamples,2))*np.array((offset, offset)) # a \in [a_true - offset/2, a_true + offset/2], lam \in [lam_true - offset/2, lam_true + offset/2]

            count = 0
            count2 = 0
            tol = 1.0
            # max_of = -1
            for params in uf.parallelize_iterable(a_lam_samples, rank, nprocs):
                try:
                    pt = np.empty(3)
                    of_val = z_system.of(params)
                    success = True
                    if of_val > tol:
                        result = minimize(z_system.of, params, method='SLSQP') #, tol=1, options={'ftol' : 1})
                        if result.fun < tol:
                            pt = (result.x[0], result.x[1], result.fun)
                        else:
                            pt = (a_true, lam_true, 0)
                        # print result.fun
                        # max_of = np.max(np.array((max_of, result.fun)))
                        # print result.fun
                    else:
                        pt = (params[0], params[1], of_val)
                        
                except (CustomErrors.IntegrationError, TypeError):
                    continue
                else:
                    data[count] = pt
                    count = count + 1
                #     if result.success:
                #         data[count] = (result.x[0], result.x[1], result.fun)
                #         count = count + 1
                # try:
                #     result = z_system.of(params)
                # except (CustomErrors.IntegrationError, TypeError):
                #     continue
                # else:
                #     if result < tol:
                #         data2[count2] = (params[0], params[1], result)
                #         count2 = count2 + 1
                
            # print rank, max_of
            
            data = data[:count]
            all_data = comm.gather(data, root=0)

            data2 = data2[:count]
            all_data2 = comm.gather(data2, root=0)

            if rank is 0:
                all_data = np.concatenate(all_data)
                all_data.dump('./data/a-lam-ofevals-2016.csv')

                all_data2 = np.concatenate(all_data2)
                all_data2.dump('./data/a-lam-ofevals-2016-2.csv')

                print '******************************'
                print 'Data saved in ./data/a-lam-ofevals-2016.csv, rerun to perform DMAP'
                print '******************************'
        
    else:
        have_x2_y2_data = True
        have_a_lam_data = False
        if have_x2_y2_data:
            print 'loading pre-existing data from ./data/x2-y2-ofevals-2016.csv (x2, y2 values from repeated optimization in transformed param space)'

            # perform the dmap on a selection of the optimization results

            # slim data, spurious results due to overflow
            data = np.load('./data/x2-y2-ofevals-2016.csv')
            # extract sloppy parameter combinations
            tol = 2.0
            x2_y2_of = data[data[:,-1] < tol]
            npts = x2_y2_of.shape[0]
            print 'Performing analysis on', npts, 'points found through optimization'

            print np.max(np.log10(x2_y2_of[:,2]))

            colornorm = colors.Normalize(vmin=np.min(np.log10(x2_y2_of[:,2])), vmax=np.max(np.log10(x2_y2_of[:,2]))) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
            colormap = cm.ScalarMappable(norm=colornorm, cmap='viridis')

            # # plot output
            scatter_size = 50
            # plot x2, y2 colored by obj. fn.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=colormap.to_rgba(np.log10(x2_y2_of[:,2])), s=scatter_size)
            axins = inset_axes(ax, 6, 5, bbox_to_anchor=(0.45, 0.9), bbox_transform=ax.transAxes)
            axins.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=colormap.to_rgba(np.log10(x2_y2_of[:,2])), s=scatter_size)
            axins.set_xlim((0, 1.2))
            axins.set_ylim((0.1, 0.3))
            axins.tick_params(axis='both', which='major', labelsize=0)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            fig.subplots_adjust(bottom=0.15)
            mark_inset(ax, axins, 1, 4 , fc='none', ec='0.5', zorder=3)

            # # investigate where points lie in (alpha, lambda) space, color by of
            lam, alpha = henon_inv(x2_y2_of[:,0], x2_y2_of[:,1], nhenon_transforms, 1.3, b)
            gsize = 40
            gspec = gs.GridSpec(gsize, gsize)
            fig = plt.figure()
            ax = fig.add_subplot(gspec[:,:37])
            cbar = ax.scatter(lam, alpha, c=np.log10(x2_y2_of[:,2]), cmap='viridis', norm=colornorm, s=scatter_size)
            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(r'$\alpha$')

            ax_cb = fig.add_subplot(gspec[:,38:])
            cb = fig.colorbar(cbar, ax_cb)
            cb.set_ticks((-1.8, -1, -.1))
            cb.set_label(r'$\log(c(\theta))$')
            fig.subplots_adjust(left=0.12, right=0.86)

            plt.show()

            # do dmaps in transformed space
            # eps = 0.6
            # k = 12
            # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps)
            # for i in range(1,5):
            #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)

            # # do dmaps in "correct" transformation (i.e. in the original alpha, lambda)
            # eps = 0.7
            # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps, metric=z_system.henon_2inv_dmaps_metric)
            # plot_dmaps.plot_xy(eigvects[:,1], eigvects[:,4], color=np.log10(x2_y2_of[:,2]), scatter=True, xlabel=r'$\Phi_1$', ylabel=r'$\Phi_4$', s=scatter_size)
            # for i in range(1,5):
            #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)
        elif have_a_lam_data:
            print 'loading pre-existing data from ./data/a-lam-ofevals-2016.csv (a, lambda values from repeated optimization in transformed param space)'

            # perform the dmap on a selection of the optimization results

            # slim data, spurious results due to overflow
            data = np.load('./data/a-lam-ofevals-2016.csv')
            # data[:,:2] = 10*data[:,:2]
            # extract sloppy parameter combinations
            tol = 1e-4
            a_lam_of = data[data[:,2] < tol]
            x2, y2 = henon(a_lam_of[:,1], a_lam_of[:,0], nhenon_transforms, a, b)
            x2_y2_of = np.array((x2,y2,a_lam_of[:,2])).T
            npts = x2_y2_of.shape[0]
            print 'Performing analysis on', npts, 'points found through optimization'

            # # plot output
            scatter_size = 100
            # plot x2, y2 colored by obj. fn.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=np.log10(x2_y2_of[:,2]), s=scatter_size, cmap='viridis')
            axins = inset_axes(ax, 6, 5, bbox_to_anchor=(0.45, 0.75), bbox_transform=ax.transAxes)
            axins.scatter(x2_y2_of[:,0], x2_y2_of[:,1], c=np.log10(x2_y2_of[:,2]), s=50, cmap='viridis')
            axins.set_xlim((-0.5, 1.25))
            axins.set_ylim((0, 0.35))
            axins.tick_params(axis='both', which='major', labelsize=0)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            fig.subplots_adjust(bottom=0.15)
            mark_inset(ax, axins, 1, 4 , fc='none', ec='0.5', zorder=3)
            plt.show()

            # # investigate where points lie in (alpha, lambda) space, color by of
            cmap = 'viridis_r'
            lam, alpha = a_lam_of[:,1], a_lam_of[:,0]
            gsize = 10
            gspec = gs.GridSpec(gsize, gsize)
            fig = plt.figure()
            ax = fig.add_subplot(gspec[:,:gsize-1])
            c = ax.scatter(lam, alpha, c=np.log10(x2_y2_of[:,2]), s=100, cmap=cmap)
            ax.set_xlim((0.99995*np.min(lam), 1.00005*np.max(lam)))
            ax.set_ylim((0.99995*np.min(alpha), 1.00005*np.max(alpha)))
            # fig.colorbar(c, format='%1.3e', ticklocation=np.linspace(np.min(np.log10(x2_y2_of[:,2])), np.max(np.log10(x2_y2_of[:,2])), 5), ticks=np.power(10, np.linspace(np.min(np.log10(x2_y2_of[:,2])), np.max(np.log10(x2_y2_of[:,2])), 5)))

            ax_cb = fig.add_subplot(gspec[:,gsize-1])
            colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colors.Normalize(np.min(np.log10(x2_y2_of[:,2])), np.max(np.log10(x2_y2_of[:,2]))), ticks=np.logspace(np.log10(np.min(np.log10(x2_y2_of[:,2]))), np.log10(np.max(np.log10(x2_y2_of[:,2]))), 3))

            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(r'$\alpha$')
            # fig.subplots_adjust(bottom=0.15)

            plt.show()

            # do dmaps in transformed space
            # eps = 0.6
            # k = 12
            # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps)
            # for i in range(1,5):
            #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)

            # # do dmaps in "correct" transformation (i.e. in the original alpha, lambda)
            # eps = 0.7
            # eigvals, eigvects = dmaps.embed_data(x2_y2_of[:,:2], k=k, epsilon=eps, metric=z_system.henon_2inv_dmaps_metric)
            # plot_dmaps.plot_xy(eigvects[:,1], eigvects[:,4], color=np.log10(x2_y2_of[:,2]), scatter=True, xlabel=r'$\Phi_1$', ylabel=r'$\Phi_4$', s=scatter_size)
            # for i in range(1,5):
            #     plot_dmaps.plot_xy(x2_y2_of[:,0], x2_y2_of[:,1], color=eigvects[:,i], scatter=True, xlabel=r'$x_2$', ylabel=r'$y_2$', s=scatter_size)
            


def comp_coeffs(coeffs, n):
    if n > 0:
        if coeffs.shape[0] - n == 1:
            coeffs[:2] = 1
        else:
            index = coeffs.shape[0] - n + 1
            temp_coeffs = np.empty(index)
            for i in range(index):
                temp_coeffs[i] = coeffs[i-1] + coeffs[i]
            coeffs[:index] = temp_coeffs
        return comp_coeffs(coeffs, n-1)
    else:
        return coeffs


def method_of_lines(L, f, n, dt):
    """Computes 'f' after 'n' steps of size 'dt', based on Laplace eqn. with Neumann BCs, could use recursion bobursion for coolness"""
    coeffs = comp_coeffs(np.zeros(n+1), n)
    print coeffs
    f_t = np.copy(f)
    Lf = np.dot(L, f_t)
    for i in range(1, n+1):
        f = f + coeffs[i]*Lf*np.power(dt, i)
        Lf = np.dot(L, Lf)
    return f
    
def discretized_laplacian_dmaps():
    """Compares the difference between a finite difference approximation of the Laplace operator with the W matrix used in DMAPS"""
    # dmaps, this assumes J = K = 0: the paper specifies J = 0, but doesn't specify what K should be represented with in this case
    n = 100
    dmaps_w = np.zeros((n,n))
    for i in range(1,n-1):
        dmaps_w[i,i+1] = 1
        dmaps_w[i,i-1] = 1
    dmaps_w[0,0] = 1
    dmaps_w[0,1] = 1
    dmaps_w[n-1,n-2] = 1
    dmaps_w[n-1,n-1] = 1
    alphas_dmaps = np.array([(np.pi*i)/n for i in range(n)])
    omegas_dmaps = (np.pi - alphas_dmaps)/2.0
    eigvals_dmaps = 2*np.cos(alphas_dmaps)
    eigvects_dmaps = np.sin(np.outer(alphas_dmaps, np.arange(1,n+1)) + np.outer(omegas_dmaps, np.ones(n))).T
    eigvects_dmaps = eigvects_dmaps/np.linalg.norm(eigvects_dmaps, axis=0)
    print 'error in dmaps eigendecomp:', np.linalg.norm(np.dot(dmaps_w, eigvects_dmaps) - np.dot(eigvects_dmaps, np.diag(eigvals_dmaps)))

    # discrete laplacian
    discrete_laplacian = np.zeros((n,n))
    for i in range(1,n-1):
        discrete_laplacian[i,i+1] = 1
        discrete_laplacian[i,i-1] = 1
    discrete_laplacian[0,1] = 2
    discrete_laplacian[n-1,n-2] = 2
    alphas_lap = np.empty(n)
    alphas_lap[:-1] = [i*np.pi/(n-1) for i in range(n-1)]
    alphas_lap[-1] = np.pi
    omegas_lap = np.empty(n)
    omegas_lap[:-1] = (np.pi - 2*alphas_lap[:-1])/2.0
    omegas_lap[-1] = np.pi/2
    eigvals_lap = 2*np.cos(alphas_lap)
    eigvects_lap = np.sin(np.outer(alphas_lap, np.arange(1,n+1)) + np.outer(omegas_lap, np.ones(n))).T
    eigvects_lap = eigvects_lap/np.linalg.norm(eigvects_lap, axis=0)
    print 'error in discrete laplacian eigendecomp:', np.linalg.norm(np.dot(discrete_laplacian, eigvects_lap) - np.dot(eigvects_lap, np.diag(eigvals_lap)))

    # # plot eigenvalues
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(n), eigvals_dmaps, color='b')
    # ax.scatter(range(n), eigvals_dmaps, label='DMAPS', color='b')
    # ax.plot(range(n), eigvals_lap, color='r')
    # ax.scatter(range(n), eigvals_lap, label='Discretization', color='r')
    # plt.show()

    # # plot differences in eigenvectors as measured by norm
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(range(n), np.linalg.norm(eigvects_lap - eigvects_dmaps, axis=0))
    # ax.scatter(range(n), np.linalg.norm(eigvects_lap - eigvects_dmaps, axis=0))
    # plt.show()
    
    # # make the actual matrices
    # dmaps
    delta = 0.1 # delta = exp(-(dx/eps)^2), ideally a small number
    W = delta*dmaps_w
    for i in range(1,n-1):
        W[i,i] = 1 - 2*delta
    W[0,0] = 1 - delta
    W[n-1,n-1] = 1 - delta
    eigvals_dmaps = delta*(eigvals_dmaps + 1/delta - 2)
    # W = W/delta

    # laplacian
    L = discrete_laplacian
    for i in range(n):
        L[i,i] = -2
    dx = 0.04
    L = L/(dx*dx)
    eigvals_lap = (eigvals_lap - 2)/(dx*dx)

    # examine result of applying these operators to 1-d data

    dt = 0.1
    npts = 100
    xmax = 4
    xs = np.linspace(0,xmax,npts)
    T0 = np.exp(xs - xmax)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, T0, c='c')
    ax.scatter(xs, method_of_lines(L, T0, 1, dt), c='b')
    ax.scatter(xs, np.dot(W, T0), c='r')
    plt.show()
    
    

class FormatAxis:
    def __init__(self, ax):
        self._ax = ax
        # dictionary of relevant functions/attributes
        self._d = {'x':{'ax':self._ax.xaxis, 'set-tick-pos':self._ax.set_xticks, 'set-tick-labels':self._ax.set_xticklabels, 'tick-pos':self._ax.xaxis.get_majorticklocs(), 'set-lims':self._ax.set_xlim},
             'y':{'ax':self._ax.yaxis, 'set-tick-pos':self._ax.set_yticks, 'set-tick-labels':self._ax.set_yticklabels, 'tick-pos':self._ax.yaxis.get_majorticklocs(), 'set-lims':self._ax.set_ylim},
             'z':{'ax':self._ax.zaxis, 'set-tick-pos':self._ax.set_zticks, 'set-tick-labels':self._ax.set_zticklabels, 'tick-pos':self._ax.get_zticks(), 'set-lims':self._ax.set_zlim}}


    def format(self, axis, data, format_string, offset=0, nticks=5, axis_stretch=0.05):
        # axis_stretch makes the axis limits slightly larger than necessary so that the labels do not overlap
        # flatten data in case it comes in as a 2d array
        data = data.flatten()
        # tick positions are constant, regardless of offset
        maxval, minval = np.min(data), np.max(data)
        increment = (maxval - minval)/(nticks-1)
        self._d[axis]['set-tick-pos']([minval + i*increment for i in range(nticks)])
        # subtract offset from data if using
        if offset != 0:
            if offset < 0:
                offset_str =  '- %1.2f' % abs(offset) # unicode dash u'\u2012'
            else:
                offset_str = '+ %1.2f' % abs(offset)
            # go through the terrible process of figuring out where to put the damn thing
            loc = {'x':0, 'y':0, 'z':0}
            for i,key in enumerate(d.keys()):
                if key is axis:
                    if axis is 'x':
                        loc[key] = np.min(self._d[key]['tick-pos']) - 0.00*(np.max(self._d[key]['tick-pos']) - np.min(self._d[key]['tick-pos']))
                    else:
                        loc[key] = np.max(self._d[key]['tick-pos']) + 0.00*(np.max(self._d[key]['tick-pos']) - np.min(self._d[key]['tick-pos']))
                else:
                    if key is 'x':
                        loc[key] = np.max(self._d[key]['tick-pos']) + 0.2*(np.max(self._d[key]['tick-pos']) - np.min(self._d[key]['tick-pos']))
                    else:
                        loc[key] = np.min(self._d[key]['tick-pos']) - 0.2*(np.max(self._d[key]['tick-pos']) - np.min(self._d[key]['tick-pos']))
            self._ax.text(loc['x'], loc['y'], loc['z'], offset_str, fontsize=12) #maxval-0.05*(maxval-minval)
            data = data - offset
        # set axis tick labels
        self._d[axis]['set-tick-labels']([format_string % (minval + i*increment) for i in range(nticks)])

        # stretch axes to prevent label overlap
        dval = maxval - minval
        minlim = minval - axis_stretch*dval
        maxlim = maxval + axis_stretch*dval
        self._d[axis]['set-lims']((minlim, maxlim))


def two_effective_one_neutral_dmaps_fig():
    """Plots the mixed kernel dmaps results of antonios' system fixing 'a' and varying 'a', 'eps' and 'lambda', thus a system with two effective and one neutral parameter. the mixed dmaps first uncovers the effective 'a', 'lambda' params. based on zagaris_model.main.py's dmaps_two_important_one_sloppy_only_data()"""
    data_dir = '/home/alexander/workspace/sloppy_models/zagaris_model/data/'
    params = np.load(data_dir + 'a-lam-eps-of-params-new.pkl')
    trajs = np.load(data_dir + 'a-lam-eps-trajs-new.pkl')
    tol = 1.5e-4 # 2e-2 for old data
    trajs = trajs[params[:,3] < tol]
    params = params[params[:,3] < tol]
    params = params[:,:3]
    print 'Have', params.shape[0], 'pts in dataset'
    # epsilons = np.logspace(-3, 1, 5)
    # dmaps.epsilon_plot(epsilons, trajs)
    epsilon = 1e-2 # from epsilon plot
    k = 12
    # eigvals, eigvects = dmaps.embed_data(trajs, k, epsilon=epsilon)
    # eigvals.dump(data_dir + 'dmaps-data-kernel-eigvals.pkl')
    # eigvects.dump(data_dir + 'dmaps-data-kernel-eigvects.pkl')
    eigvals = np.load(data_dir + 'dmaps-data-kernel-eigvals.pkl')
    eigvects = np.load(data_dir + 'dmaps-data-kernel-eigvects.pkl')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,1])
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(np.log10(params[:,0]), np.log10(params[:,1]), np.log10(params[:,2]), c=eigvects[:,2])
    for ax in [ax1, ax2]:
        # prevent overly crowded axis ticks and labels
        formatter = FormatAxis(ax)
        formatter.format('x', np.log10(params[:,0]), '%1.3f', nticks=2)
        formatter.format('y', np.log10(params[:,1]), '%1.3f', nticks=2)
        formatter.format('z', np.log10(params[:,2]), '%1.1f')
        # labels
        ax.set_xlabel(r'$a$')
        ax.set_ylabel(r'$\lambda$')
        ax.set_zlabel(r'$\epsilon$')
        # white bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # move labels to avoid overlap with numbers
        ax.xaxis._axinfo['label']['space_factor'] = 2.2
        ax.yaxis._axinfo['label']['space_factor'] = 2.2
        ax.zaxis._axinfo['label']['space_factor'] = 2.2
        # # offset size
        # ax.yaxis.get_children()[1].set_size(20)
    plt.show()


def calc_tri_avg(triangles, data):
    """Calculates the average value of 'data' at each triangle specified in 'triangles'.

    ***'triangles' and 'data' must be of the same length***"""
    avg_data = np.empty(triangles.shape[0])
    for i, t in enumerate(triangles):
        avg_data[i] = np.average(data[t])
    return avg_data

def rawlings_3d_dmaps_fig():
    """Plots **thick** sloppy manifold in parameter space (here found through sampling), which is colored by the first and second DMAP eigenvectors in two separate subfigures on the right, adapted from rawlings_model.main's dmaps_param_set()"""    
    paramdata = np.genfromtxt('/home/alexander/workspace/sloppy_models/rawlings_model/data/data-dmaps-params.csv', delimiter=',')
    keff = paramdata[:,0]*paramdata[:,2]/(paramdata[:,1] + paramdata[:,2])
    paramdata = np.log10(paramdata)
    eigvects = np.genfromtxt('/home/alexander/workspace/sloppy_models/rawlings_model/data/data-dmaps-eigvects.csv', delimiter=',')
    # color params by dmaps coord
    gspec = gs.GridSpec(2,2)
    fig = plt.figure()
    # set up axes
    ax_b = fig.add_subplot(gspec[:,0], projection='3d')
    ax_keff = fig.add_subplot(gspec[0,1], projection='3d')
    ax_dmaps = fig.add_subplot(gspec[1,1], projection='3d')
    # add scatter plots with correct colorings
    ax_b.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c='b', cmap='jet')
    ax_keff.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=keff, cmap='jet')
    ax_dmaps.scatter(paramdata[:,0], paramdata[:,1], paramdata[:,2], c=eigvects[:,1], cmap='jet')
    for ax in [ax_b, ax_keff, ax_dmaps]:
        # label axes
        ax.set_xlabel('\n\nlog(' + r'$k_1$' + ')')
        ax.set_ylabel('\n\nlog(' + r'$k_{-1}$' + ')')
        ax.set_zlabel('\n\nlog(' + r'$k_2$' + ')')
        # move labels to avoid overlap with numbers
        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 2.8
        # white bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # no grid
        # ax.grid(False)
        # get those ticklabels outta here
        formatter = FormatAxis(ax)
        formatter.format('x', paramdata[:,0], '%1.1f', nticks=3)
        formatter.format('y', paramdata[:,1], '%1.1f', nticks=3)
        formatter.format('z', paramdata[:,2], '%1.1f', nticks=3)

    plt.show()

def rawlings_surface_normal():
    """Uses PCA to calculate the direction normal to a patch of the keff = c surface, used to verify this direction actually corresponds to changing keff"""
    data = np.load('../rawlings_model/data/params-ofevals.pkl')
    # k1_max = 10
    # kinv_min = 100
    # k2_min = 100
    # data = data[data[:,1] < k1_max]
    # data = data[data[:,2] > kinv_min]
    # data = data[data[:,3] > k2_min]
    of_max = 1e-3
    data = data[data[:,0] < of_max]
    print 'shape of data after trimmming:', data.shape

    log_params = np.log10(data[:,1:])
    delta = 0.2
    plane_region = np.empty((log_params.shape[0],3))
    center = log_params[0] # 2 random choice of center for ball
    count = 0
    for pt in log_params:
        if np.linalg.norm(pt - center) < delta:
            plane_region[count] = pt
            count = count + 1
    print 'have', count, 'pts in ball'
    print 'centered at', center
    plane_region = plane_region[:count]
    keff_pr = plane_region[:,0]*plane_region[:,2]/(plane_region[:,1] + plane_region[:,2])
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    k1s = data[:,1];
    kinvs = data[:,2];
    k2s = data[:,3]
    keff2 = (k1s - kinvs)*k1s/np.power(kinvs + k2s, 2)
    

    # visualize region
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(plane_region[:,0], plane_region[:,1], plane_region[:,2])
    slice = 50
    ax.scatter(log_params[::slice,0], log_params[::slice,1], log_params[::slice,2], c=np.log10(data[::slice,0]))
    ax.scatter(center[0], center[1], center[2], s=100, lw=20)
    formatter = FormatAxis(ax)
    formatter.format('x', log_params[::slice,0], '%1.1f', nticks=3)
    formatter.format('y', log_params[::slice,1], '%1.1f', nticks=3)
    formatter.format('z', log_params[::slice,2], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')
    plt.show()

    pcs, vars = pca(plane_region, k=3)
    normal = pcs[:,2]/np.linalg.norm(pcs[:,2])
    k1 = np.power(10, center[0])
    kinv = np.power(10, center[1])
    k2 = np.power(10, center[2])
    keff_gradient = np.array((k2/(kinv + k2), -k1*k2/np.power(kinv + k2, 2), k1/(kinv + k2) - k1*k2/np.power(kinv + k2, 2)))
    keff_loggradient = np.array((k1*k2/(kinv + k2), -k1*k2*kinv/np.power(kinv + k2, 2), k1*k2/(kinv + k2) - k1*k2*k2/np.power(kinv + k2, 2)))
    keff_loggradient = keff_loggradient/np.linalg.norm(keff_loggradient)
    print '******************************'
    print 's1/s3, s2/s3:', vars[:2]/vars[2]
    print 'normal:', normal
    print 'keff grad:', keff_loggradient
    print '| normal - keff_grad |', np.linalg.norm(normal - keff_loggradient)
    print 'keff max, min:', np.max(keff), np.min(keff)
    


def rawlings_2d_dmaps_fig():
    """Plots sloppy manifold in parameter space (here found through sampling), which is colored by the first and second DMAP eigenvectors in two separate subfigures on the right, adapted from rawlings_model.main's dmaps_param_set()"""
    # import data
    # eigvals.dump('./data/rawlings-2d-dmaps-eigvals.pkl')
    # eigvects.dump('./data/rawlings-2d-dmaps-eigvects.pkl')
    # log_params_data.dump('./data/rawlings-2d-dmaps-logparams.pkl')
    params_data = np.load('./data/params-ofevals-slimmed.pkl')
    
    log_params_data = np.log10(params_data[:,1:]) # np.load('./data/rawlings-2d-dmaps-logparams.pkl')
    
    eigvals = np.genfromtxt('./data/dmaps-eigvals--tol-0.001-k-12.csv', delimiter=',')
    eigvects = np.genfromtxt('./data/dmaps-eigvects--tol-0.001-k-12.csv', delimiter=',')

    # # plot figure (three subfigs total)
    gspec = gs.GridSpec(2,2)
    fig = plt.figure()
    xlabel = '\n\n' + r'$k_1$'
    ylabel = '\n\n' + r'$k_{-1}$'
    zlabel = '\n\n' + r'$k_2$'
    stride = 15 # effectively thins array that is plotted
    cmap = cm.ScalarMappable(cmap='jet')
    # create triangulation in k_2, k_{-1}
    triangles = tri.Triangulation(log_params_data[::stride,2], log_params_data[::stride,1]).triangles # 2, 1 was found to be best
    triangle_vertices = np.array([np.array((log_params_data[::stride][T[0]], log_params_data[::stride][T[1]], log_params_data[::stride][T[2]])) for T in triangles]) # plot in k1, k-1, k2
    # create polygon collections that will be plotted (seems impossible to completely erase edges)
    collb = Poly3DCollection(triangle_vertices, facecolors='b', linewidths=0, edgecolors='b')
    coll1 = Poly3DCollection(triangle_vertices, facecolors=cmap.to_rgba(calc_tri_avg(triangles, eigvects[::stride,1])), linewidths=0, edgecolors=np.array((0,0,0,0)))
    coll2 = Poly3DCollection(triangle_vertices, facecolors=cmap.to_rgba(calc_tri_avg(triangles, eigvects[::stride,2])), linewidths=0, edgecolors=np.array((0,0,0,0)))
    # create axes for each subplot
    ax_b = fig.add_subplot(gspec[:,0], projection='3d')
    ax_d1 = fig.add_subplot(gspec[0,1], projection='3d')
    ax_d2 = fig.add_subplot(gspec[1,1], projection='3d')
    # add collections to axes
    ax_b.add_collection(collb) # plain blue dataset
    ax_d1.add_collection(coll1) # colored by dmaps1
    ax_d2.add_collection(coll2) # colored by dmaps2
    # add labels and otherwise adjust plots
    axes = [ax_b, ax_d1, ax_d2]
    for ax in axes:
        # axis limits
        ax.set_xlim((0.99*np.min(log_params_data[::stride,0]), 1.01*np.max(log_params_data[::stride,0])))
        ax.set_ylim((0.99*np.min(log_params_data[::stride,1]), 1.01*np.max(log_params_data[::stride,1])))
        ax.set_zlim((0.99*np.min(log_params_data[::stride,2]), 1.01*np.max(log_params_data[::stride,2])))
        # axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # hide labels, too squashed
        formatter = FormatAxis(ax)
        formatter.format('x', log_params_data[:,0], '%1.1f', nticks=3)
        formatter.format('y', log_params_data[:,1], '%1.1f', nticks=3)
        formatter.format('z', log_params_data[:,2], '%1.1f', nticks=3)
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # # hide grid
        # ax.grid(False)
    plt.tight_layout()
    plt.show()

def dmaps_2d_epsilon():
    """Examines the effect of epsilon on the embedding's eigenvalues"""
    # TODO: change the wonky indexing. In some places you remove the trivial eigenvalue but not the corresponding eigenvector
    # calculate dmap on [a,b] with relatively small epsilon
    # a = 2e-3
    # b = 1e-3
    # npts = 100
    # dmaps_epsilon = 1e-3 # dmaps uses eps^2, theory uses eps
    # theory_epsilon = dmaps_epsilon*dmaps_epsilon
    a = 4
    b = 2
    npts_xaxis = 100
    npts_yaxis = 40
    npts = npts_xaxis*npts_yaxis
    dmaps_epsilon = 0.4
    theory_epsilon = dmaps_epsilon*dmaps_epsilon
    # xdata, ydata = np.meshgrid(a*np.random.uniform(size=npts_xaxis), b*np.random.uniform(size=npts_yaxis)) # np.meshgrid(np.linspace(0,a,npts_xaxis), np.linspace(0,b,npts_yaxis))
    # xdata = xdata.flatten()
    # ydata = ydata.flatten()
    xdata = a*np.random.uniform(size=npts)
    ydata = b*np.random.uniform(size=npts)
    data = np.empty((npts,2))
    data[:,0] = xdata
    data[:,1] = ydata
    k = 12
    # don't always calculate
    dmaps_eigvals, eigvects = dmaps.embed_data(data, k, epsilon=dmaps_epsilon, embedding_method=dmaps._compute_embedding_laplace_beltrami)
    dmaps_eigvals.dump('./data/temp-eigvals2.pkl')
    eigvects.dump('./data/temp-eigvects2.pkl')
    # dmaps_eigvals = np.load('./data/temp-eigvals2.pkl')
    # eigvects = np.load('./data/temp-eigvects2.pkl')
    # calculate theoretical eigenvalues and raise to e^{-eigval*theory_eps}, hopefully match dmaps eps
    theory_eigvals = np.empty(36)
    count = 0
    for i in range(6):
        for j in range(6):
            theory_eigvals[count] = np.power(i*np.pi/a, 2) + np.power(j*np.pi/b, 2)
            count = count + 1
    # sort them in increasing order, discard eigval=0
    theory_eigvals_argsort = np.argsort(theory_eigvals)
    theory_eigvals = theory_eigvals[theory_eigvals_argsort]
    # random scaling factor from operator_0 to operator (pg 28 lafon's thesis)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # only valid for density of 2500 pts per [0,0.2] x [0,0.2] square
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot exp(-eps*theory_eigvals) vs dmaps_eigvals
    scaling_factor = 4.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1,k), dmaps_eigvals[1:k], color='r', label='DMAPS', lw=4)
    ax.scatter(range(1,k), np.exp(-theory_epsilon*theory_eigvals[1:k]/scaling_factor), c='b', label='Analytical', s=100)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\lambda$')
    ax.legend()
    plt.savefig('./figs/analytical-dmaps-comparison/eigvals-comparison.png')

    # examine eigenfunctions vs eigenvectors
    theory_eigenfunctions = np.empty((36, npts))
    xs = np.linspace(0,a,npts_xaxis)
    ys = np.linspace(0,b,npts_yaxis)
    count = 0
    for i in range(6):
        for j in range(6):
            theory_eigenfunctions[count] = np.cos(i*data[:,0]*np.pi/a)*np.cos(j*data[:,1]*np.pi/b)
            count = count + 1
    theory_eigenfunctions = theory_eigenfunctions[theory_eigvals_argsort]
    xgrid, ygrid = np.meshgrid(xs, ys)
    for i in range(1,k):
        # attempt to get consistent coloring by swapping sign of eigvector if necessary
        if np.sign(eigvects[0,i]) != np.sign(theory_eigenfunctions[i,0]):
            eigvects[:,i] = -1*eigvects[:,i]
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(data[:,0], data[:,1], c=eigvects[:,i], s=100)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('DMAPS')
        fig1.tight_layout()
        plt.savefig('./figs/analytical-dmaps-comparison/eigvect' + str(i) + '.png')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(data[:,0], data[:,1], c=theory_eigenfunctions[i], s=100)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Analytical')
        fig2.tight_layout()
        plt.savefig('./figs/analytical-dmaps-comparison/eigfn' + str(i) + '.png')
        

    # show quantitative error between eigfns and eigvects
    relative_error = np.empty(k-1)
    for i in range(1,k):
        scale = np.linalg.norm(theory_eigenfunctions[i])
        relative_error[i-1] = np.linalg.norm(scale*eigvects[:,i] - theory_eigenfunctions[i])/scale
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(1,k), relative_error, s=100)
    ax.set_xlabel(r'$\Phi_i$')
    ax.set_ylabel('relative error in DMAP')
    ax.set_ylim(bottom=0)
    plt.savefig('./figs/analytical-dmaps-comparison/eigvects-error.png')
        
    

class diffusion_sln:

    def __init__(self, imax, xmax, ymax, x0, y0, D, epsilon):
        self._indices = np.arange(1,imax+1)
        self._indices.shape = (imax,1)
        self._ymax = ymax
        self._xmax = xmax
        self._x0 = x0
        self._y0 = y0
        self._A = 4.0*(np.sin(self._indices*np.pi*x0/xmax)*np.sin(self._indices*np.pi*y0/ymax).T)/(ymax*xmax)
        self._E = np.exp(-np.power(D*np.pi*self._indices/xmax, 2))*np.exp(-np.power(epsilon*D*np.pi*self._indices/ymax, 2)).T

    def __call__(self,x,y,t):
        siny = np.sin(self._indices*np.pi*y/self._ymax)
        sinx = np.sin(self._indices*np.pi*x/self._xmax)
        E_t = np.power(self._E, t)
        
        return np.dot(siny.T, np.dot(self._A*E_t, sinx))

def analytical_anisotropic_diffusion():
    """Plots the results of two-dimensionnal anisotropic diffusion on a square"""
    imax = 10
    xmax = 3
    ymax = 2
    x0 = 1
    y0 = 1.5
    D = 1
    epsilon = 0.5
    diffusion_f = diffusion_sln(imax, xmax, ymax, x0, y0, D, epsilon)
    t = 1
    npts = 100
    sln = np.empty((npts,npts))
    xs = np.linspace(0,xmax,npts)
    ys = np.linspace(0,ymax,npts)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            sln[i,j] = diffusion_f(x, y, t)
    xgrid, ygrid = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xgrid, ygrid, c=sln)
    plt.show()

def analytical_anisotropic_diffusion_eigenfns():
    """Plots two-dimensional eigenfunctions of anisotropic diffusion operator"""
    # create indices sorted by increasing corresponding eigenvalue
    xmax = 1.0
    ymax = 1.0
    epsilon = 1e-1
    imax = 100
    # ivals, jvals = np.meshgrid(range(1,imax+1), range(1,imax+1))
    ivals, jvals = np.meshgrid(range(imax), range(imax))
    eigvals = np.power(ivals/xmax, 2) + epsilon*np.power(jvals/ymax, 2)
    sorted_eigval_indices = np.argsort(eigvals.flatten())
    eigvals = eigvals.flatten()[sorted_eigval_indices]
    ivals = ivals.flatten()[sorted_eigval_indices]
    jvals = jvals.flatten()[sorted_eigval_indices]
    # create grid for sampling fn
    npts = 100
    xgrid, ygrid = np.meshgrid(np.linspace(0,xmax,npts), np.linspace(0,ymax,npts))
    # plot first k eigenfns
    k = 40
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(False)
    for i in range(k):
        # f = np.sin(ivals[i]*np.pi*xgrid/xmax)*np.sin(jvals[i]*np.pi*ygrid/ymax)
        f = np.cos(ivals[i]*np.pi*xgrid/xmax)*np.cos(jvals[i]*np.pi*ygrid/ymax)
        ax.scatter(xgrid, ygrid, c=f)
        ax.set_title(r'$\lambda=$' + str(eigvals[i]))
        plt.savefig('./figs/dmaps/diffusion-eigenfn' + str(i) + '.png')


def dmaps_plane():
    """Tests whether analytical results for anisotropic diffusion on a plane match numerical results"""
    npts = 50
    xdata, ydata = np.meshgrid(np.linspace(0,1,npts), np.linspace(0,1,npts))
    xdata = xdata.flatten()
    ydata = ydata.flatten()
    data = zip(xdata,ydata)
    epsilon = 1e-1
    alpha = 1.0
    kernel = data_kernel(epsilon, alpha)
    k = 14
    eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel, symmetric=True)
    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(False)
    for i in range(1,k):
        ax.scatter(xdata, ydata, c=eigvects[:,i], cmap='jet', s=50)
        plt.savefig('./figs/dmaps/dmaps-plane-' + str(i) + '.png')


def dmaps_1d_dataspace():
    """Examines the result of different weightings on the DMAP of the combination of some function f:R2 -> R1, where the kernel is a weighted data-space kernel"""
    # sample (x,y) deterministically on (0,1)^2
    sqrt_npts = 50
    npts = sqrt_npts*sqrt_npts
    x = np.linspace(0,1,sqrt_npts)
    y = np.linspace(0,1,sqrt_npts)
    xgrid, ygrid = np.meshgrid(x,y)
    domain_data = np.array((xgrid.flatten(), ygrid.flatten())).T # shape = (npts,2)
    # simply take f = x + y
    fdata = np.sum(domain_data, axis=1)/2.0
    # combine preimage and image to make complete dataset
    dataset = zip(domain_data, fdata)
    # # plot dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(domain_data[:,0], domain_data[:,1], fdata)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel(r'$f(x,y)$')
    # plt.show(fig)
    # # examine epsilons affect on domain and range separately
    # neps = 6
    # epsilons = np.logspace(-4, 0, neps)
    # dmaps.epsilon_plot(epsilons, domain_data, filename='./figs/domaindata-eps-plot.png')
    # dmaps.epsilon_plot(epsilons, fdata, filename='./figs/fdata-eps-plot.png')
    # eps { 0.01 -> 0 looks reasonalbe
    epsilon = 1e-1
    alpha = 4 # 1e-8
    kernel = data_kernel(epsilon, alpha)
    k = 24
    eigvals, eigvects = dmaps.embed_data_customkernel(dataset, k, kernel)
    fig = plt.figure()
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    ax3d.hold(False)
    ax2d.hold(False)
    for i in range(1,k):
        ax3d.scatter(domain_data[:,0], domain_data[:,1], fdata, c=eigvects[:,i])
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel(r'$f(x,y)$')
        ax2d.scatter(fdata, np.ones(npts), c=eigvects[:,i])
        plt.savefig('./figs/dmaps/dmaps-1d-dataspace-alpha-' + str(alpha) + '-eps-' + str(epsilon) + '-' + str(i) + '.png')


def dmaps_2d_dataspace():
    """Examines the result of different weightings on the DMAP of the combination of some function f:R2 -> R2, where the kernel is a weighted data-space kernel"""
    # sample (x,y) deterministically on (0,1)^2
    sqrt_npts = 70
    npts = sqrt_npts*sqrt_npts
    x = np.linspace(0,1,sqrt_npts)
    y = np.linspace(0,1,sqrt_npts)
    xgrid, ygrid = np.meshgrid(x,y)
    domain_data = np.array((xgrid.flatten(), ygrid.flatten())).T # shape = (npts,2)
    # set up transformation matrix
    lam1 = np.sqrt(0.5)
    lam2 = np.sqrt(0.5)
    theta1 = 0
    theta2 = theta1 + np.pi/2 # creates orthogonal eigenvectors
    V = np.array(((np.cos(theta1), np.cos(theta2)), (np.sin(theta1), np.sin(theta2)))) # orthonormal basis for R2
    L = np.diag(np.array((lam1, lam2))) # diagonal eigenvalue matrix
    A = np.dot(V, np.dot(L, np.linalg.inv(V)))
    print A
    # f = lambda x: np.dot(A, x)
    fdata = np.dot(A, domain_data.T).T
    # # # plot dataset
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(domain_data[:,0], domain_data[:,1], c='b', s=40)
    # ax.scatter(fdata[:,0], fdata[:,1], c='r')
    # plt.show()
    # # examine epsilons affect on domain and range separately
    # neps = 6
    # epsilons = np.logspace(-4, 0, neps)
    # dmaps.epsilon_plot(epsilons, domain_data, filename='./figs/domaindata-eps-plot.png')
    # dmaps.epsilon_plot(epsilons, fdata, filename='./figs/fdata-eps-plot.png')

    # data = zip(domain_data, fdata)
    # epsilon = 0.1 # 1e-1
    # alpha = 10 # 1e-8
    # kernel = data_kernel(epsilon, alpha)
    # k = 24
    # eigvals, eigvects = dmaps.embed_data_customkernel(data, k, kernel)

    # for antonios
    alpha = 10
    epsilon = 0.1
    k = 24
    data = np.empty((npts, 4))
    data[:,:2] = domain_data
    data[:,2:] = fdata
    eigvals, eigvects = dmaps.embed_data(data, k, epsilon=epsilon)

    fig = plt.figure()
    axp = fig.add_subplot(121) # preimage
    axi = fig.add_subplot(122) # image
    axp.hold(False)
    axi.hold(False)
    for i in range(1,k):
        axp.scatter(domain_data[:,0], domain_data[:,1], c=eigvects[:,i])
        axp.set_title('preimage')
        axi.scatter(1 + fdata[:,0], fdata[:,1], c=eigvects[:,i])
        axi.set_title('image')
        plt.savefig('./figs/dmaps/dmaps-2d-dataspace-lam2-' + str(lam2) + '-alpha-'+ str(alpha) + '-eps-' + str(epsilon) + '-' + str(i) + '.png')


def dmaps_line():
    """Performs DMAPS on a one-dimensional dataset at different values of epsilon to investigate this parameter's effect on the resulting embedding"""
    # create dataset on line, no randomness to prevent run-to-run variability
    npts = 500
    dataset = np.linspace(0,1,npts) # np.sort(np.random.uniform(size=npts))
    # create set of epsilons to test
    neps = 40
    epsilons = np.logspace(-4, -3, neps) # np.logspace(-4, 0, neps)
    k = 5
    # space to save dmaps results
    embeddings = np.empty((neps, npts))
    eigvals_save = np.empty(neps)
    for i, eps in enumerate(epsilons):
        progress_bar(i, neps)
        eigvals, eigvects = dmaps.embed_data(dataset, k, epsilon=eps)
        embeddings[i] = eigvals[1]*eigvects[:,1]
        eigvals_save[i] = eigvals[1]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # real adhoc test to get colorings to line up properly later on
        # some of the eigenvectors are flipped compared to others, so we unflip them for consistency
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if embeddings[i,0] > embeddings[i,-1]:
            embeddings[i] = embeddings[i,::-1]
    # fig setup, including consistent colornorm for all results to compare across different epsilons
    emax = np.max(np.abs(embeddings))
    colornorm = colors.Normalize(vmin=-emax, vmax=emax) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='PuOr')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    for i in range(neps):
        ax.scatter(dataset, epsilons[i]*np.ones(npts), c=colormap.to_rgba(embeddings[i]))
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_ylim((0.9*np.min(epsilons), 1.1*np.max(epsilons)))
    ax.set_yscale('log')
    plt.savefig('./figs/1d-dmaps' + str(i) + '.png')
    print eigvals_save
        
class Singularly_Perturbed_System:

    def __init__(self, eps_true, y0_true, of_val, times):
        self._lam = 2.0
        self._x0 = 1.0
        self._of_val = of_val
        self._times = times
        self._traj = self._get_traj(eps_true, y0_true)

    def _get_traj(self, eps, y0):
        f = lambda t, x: np.dot(np.array(((-self._lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
        integrator = Integration.Integrator(f)
        traj = integrator.integrate(np.array((self._x0,y0)), self._times)[:,1]
        return traj

    def change_of_val(self, new_of_val):
        self._of_val = new_of_val

    def get_f_val(self, eps, y0):
        """returns value of f = c(eps, y0) - of_val"""
        return np.array((np.linalg.norm(self._get_traj(eps, y0) - self._traj, 2) - self._of_val,))

    def _get_f_val(self, params):
        """returns value of f = c(eps, y0) - of_val, but takes argument as np array, only used internally by 'get_Df_val'"""
        eps, y0 = params
        return np.linalg.norm(self._get_traj(eps, y0) - self._traj, 2) - self._of_val

    def get_Df_val(self, eps, y0):
        """Returns 1x2 array [df_deps, df_dy0]"""
        grad = gradient(self._get_f_val, np.array((eps, y0)), h=1e-5)
        grad.shape = (1, grad.shape[0])
        return grad
        
class Inverse_Singularly_Perturbed_System:

    def __init__(self, epsinv_true, y0_true, of_val, times):
        self._lam = 2.0
        self._x0 = 1.0
        self._of_val = of_val
        self._times = times
        self._traj = self._get_traj(epsinv_true, y0_true)

    def _get_traj(self, epsinv, y0):
        f = lambda t, x: np.dot(np.array(((-self._lam, 1),(1,(-epsinv)*(1+10/(1.5-np.sin(x[1])))))), x)
        integrator = Integration.Integrator(f)
        traj = integrator.integrate(np.array((self._x0,y0)), self._times)[:,1]
        return traj

    def change_of_val(self, new_of_val):
        self._of_val = new_of_val

    def get_f_val(self, epsinv, y0):
        """returns value of f = c(eps, y0) - of_val"""
        return np.array((np.linalg.norm(self._get_traj(epsinv, y0) - self._traj, 2) - self._of_val,))

    def _get_f_val(self, params):
        """returns value of f = c(eps, y0) - of_val, but takes argument as np array, only used internally by 'get_Df_val'"""
        return self.get_f_val(params[0], params[1])

    def get_Df_val(self, epsinv, y0):
        """Returns 1x2 array [df_deps, df_dy0]"""
        grad = gradient(self._get_f_val, np.array((epsinv, y0)), h=1e-8)
        grad.shape = (1, grad.shape[0])
        return grad


class Log_Singularly_Perturbed_System:

    def __init__(self, log_eps_true, y0_true, of_val, times):
        self._lam = 2.0
        self._x0 = 1.0
        self._of_val = of_val
        self._times = times
        self._traj = self._get_traj(log_eps_true, y0_true)

    def _get_traj(self, log_eps, y0):
        f = lambda t, x: np.dot(np.array(((-self._lam, 1),(1,(-1/np.power(10, log_eps))*(1+10/(1.5-np.sin(x[1])))))), x)
        integrator = Integration.Integrator(f)
        traj = integrator.integrate(np.array((self._x0,y0)), self._times)[:,1]
        return traj

    def change_of_val(self, new_of_val):
        self._of_val = new_of_val

    def get_f_val(self, log_eps, y0):
        """returns value of f = c(log_eps, y0) - of_val"""
        return np.array((np.linalg.norm(self._get_traj(log_eps, y0) - self._traj, 2) - self._of_val,))

    def _get_f_val(self, params):
        """returns value of f = c(log_eps, y0) - of_val, but takes argument as np array, only used internally by 'get_Df_val'"""
        log_eps, y0 = params
        return np.linalg.norm(self._get_traj(log_eps, y0) - self._traj, 2) - self._of_val

    def get_Df_val(self, log_eps, y0):
        """Returns 1x2 array [df_dlog_eps, df_dy0]"""
        grad = gradient(self._get_f_val, np.array((log_eps, y0)), h=1e-12)
        grad.shape = (1, grad.shape[0])
        return grad


def sing_pert_contours_pydstool():
    """Uses PyDSTool to continue along level-sets"""
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.array((0.01, 0.1, 0.5))
    # # working param set
    # y0_true = 5
    # eps_true = 0.5
    # of_val = 0.1
    # ds = 1e-3
    # nsteps = 2000
    # set up base system used by all of_vals
    y0_true = 5.0
    epsinv_true = 3.0 # eps_true = 0.01
    of_val = 0.1
    sp_system = Inverse_Singularly_Perturbed_System(epsinv_true, y0_true, of_val, times)

    # the following are pre-computed points on the branch with of_val = 0.1, for which f(epsinv_guess, y0_guess) = 1e-13
    y0_guess = 5.02738659854498859403
    epsinv_guess = 2.74346589102169380325

    print sp_system.get_f_val(epsinv_guess, y0_guess)

    DSargs = pc.common.args(name='SPSys')
    # DSargs.pars = {'epsinv': 101, 'y0': 4.9}
    DSargs.pars = {'epsinv': epsinv_guess}
    DSargs.varspecs = {'y0': 'of_rhs_val'}
    DSargs.vfcodeinsert_start = 'of_rhs_val = ds.of_rhs(epsinv, y0)'
    DSargs.ignorespecial = ['of_rhs_val']
    DSargs.ics = {'y0': y0_guess}
    ode = pc.Generator.Vode_ODEsystem(DSargs)
    ode.of_rhs = sp_system.get_f_val

    PyCont = pc.ContClass(ode)
    PCargs = pc.common.args(name='SPcont', type='EP-C')
    PCargs.freepars = ['epsinv']
    PCargs.StepSize = 1e-8
    PCargs.MaxNumPoints = 100
    PCargs.MaxStepSize = 5e-2
    PyCont.newCurve(PCargs)

    PyCont['SPcont'].forward()

    curve = PyCont['SPcont'].sol

    print 'found', len(curve), 'pts'

def sing_pert_contours_fig1():
    """Plots contours of system used in 'sing_pert_data_space' in y0 and epsilon using arclength continuation"""
    x0 = 1
    lam = 1
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.array((0.01, 0.1, 0.5))
    # # working param set
    # y0_true = 5
    # eps_true = 0.5
    # of_val = 0.1
    # ds = 1e-3
    # nsteps = 2000
    # set up base system used by all of_vals
    y0_true = 5
    epsinv_true = 100 # eps_true = 0.01
    of_val = 0.1
    sp_system = Inverse_Singularly_Perturbed_System(epsinv_true, y0_true, of_val, times)
    # continue along y0
    psa_solver = PSA(sp_system.get_f_val, sp_system.get_Df_val)

    # fig to which every branch will be added
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lw = 2.0

    # # working param set one
    # # produces nose at larger of_val
    of_val = 0.1
    sp_system.change_of_val(of_val)
    # start solver three times at different points to flesh out the complete contour
    # (not stable enough to make it around the full ellipse in a computationally feasible time)
    ds = 1e-1
    nsteps = 2500

    for i in range(2):
        # # continue starting from three distinct points, combine results in plot
        # if i == 0:
        #     y0_guess = -0.5
        #     epsinv_guess = 0.7
        #     # y0_guess = 3.5
        #     # epsinv_guess = 120.2
        # if i == 1:
        #     y0_guess = 0
        #     epsinv_guess = 1
        # if i == 2:
        #     y0_guess = 4.77078
        #     epsinv_guess = 99.28641
        # try:
        #     branch = psa_solver.find_branch(np.array((epsinv_guess,)), y0_guess, ds, nsteps, progress_bar=True, abstol=1e-9, reltol=1e-9)
        # except CustomErrors.PSAError as e:
        #     print e.msg
        #     exit()
        # print 'found ' + str(100.0*branch.shape[0]/nsteps) + '% of pts on the level set'
        # branch.dump('./data/b1' + str(i) + '.pkl')

        # do not re-generate data
        print 'loading data for contour at of val:', of_val

        branch = np.load('./data/b1' + str(i) + '.pkl')

        label = None
        # only label once
        if i == 0:
            label = r'$c(\theta) = 10^{-1}$'
        ax.plot(branch[:,0], branch[:,1], c='m', label=label, lw=lw)

    of_val = 0.01
    sp_system.change_of_val(of_val)
    # start solver three times at different points to flesh out the complete contour
    # (not stable enough to make it around the full ellipse in a computationally feasible time)
    ds = 1e-1
    nsteps = 2500


    for i in range(2):
        # # continue starting from three distinct points, combine results in plot
        # if i == 0:
        #     y0_guess = -0.5
        #     epsinv_guess = 0.7
        # if i == 1:
        #     y0_guess = 0
        #     epsinv_guess = 1
        # if i == 2:
        #     y0_guess = 4.77078
        #     epsinv_guess = 99.28641
        # try:
        #     branch = psa_solver.find_branch(np.array((epsinv_guess,)), y0_guess, ds, nsteps, progress_bar=True, abstol=1e-9, reltol=1e-9)
        # except CustomErrors.PSAError as e:
        #     print e.msg
        #     exit()
        # print 'found ' + str(100.0*branch.shape[0]/nsteps) + '% of pts on the level set'
        # branch.dump('./data/b2' + str(i) + '.pkl')

        # do not re-generate data
        print 'loading data for contour at of val:', of_val
        branch = np.load('./data/b2' + str(i) + '.pkl')

        label = None
        # only label once
        if i == 0:
            label = r'$c(\theta) = 10^{-2}$'
        ax.plot(branch[:,0], branch[:,1], c='b', label=label, lw=lw)


    of_val = 0.001
    sp_system.change_of_val(of_val)
    # start solver three times at different points to flesh out the complete contour
    # (not stable enough to make it around the full ellipse in a computationally feasible time)
    ds = 1e-1
    nsteps = 1000
    for i in range(4):
        # # continue starting from three distinct points, combine results in plot
        # if i == 0:
        #     y0_guess = -0.005
        #     epsinv_guess = 54.02
        # if i == 1:
        #     y0_guess = -0.015
        #     epsinv_guess = 54.02
        # if i == 2:
        #     y0_guess = 4.96
        #     epsinv_guess = 90.89
        # if i == 3:
        #     y0_guess = 18.19
        #     epsinv_guess = 133.87
        # try:
        #     branch = psa_solver.find_branch(np.array((epsinv_guess,)), y0_guess, ds, nsteps, progress_bar=True, abstol=1e-9, reltol=1e-9)
        # except CustomErrors.PSAError as e:
        #     print e.msg
        #     exit()
        # print 'found ' + str(100.0*branch.shape[0]/nsteps) + '% of pts on the level set'
        # branch.dump('./data/b3' + str(i) + '.pkl')

        # do not re-generate data
        print 'loading data for contour at of val:', of_val
        branch = np.load('./data/b3' + str(i) + '.pkl')

        label = None
        # only label once
        if i == 0:
            label = r'$c(\theta) = 10^{-3}$'
        ax.plot(branch[:,0], branch[:,1], c='g', label=label, lw=lw)


    of_val = 0.0001
    sp_system.change_of_val(of_val)
    # start solver three times at different points to flesh out the complete contour
    # (not stable enough to make it around the full ellipse in a computationally feasible time)
    ds = 1e-2
    nsteps = 250

    for i in range(3):
        # continue starting from three distinct points, combine results in plot
        if i == 0:
            y0_guess = 5
            epsinv_guess = 105
        if i == 1:
            y0_guess = 5.040
            epsinv_guess = 100.61
        if i == 2:
            y0_guess = 4.77078
            epsinv_guess = 99.28641
        try:
            branch = psa_solver.find_branch(np.array((epsinv_guess,)), y0_guess, ds, nsteps, progress_bar=True, abstol=1e-9, reltol=1e-9)
        except CustomErrors.PSAError as e:
            print e.msg
            exit()
        print 'found ' + str(100.0*branch.shape[0]/nsteps) + '% of pts on the level set'
        branch.dump('./data/b4' + str(i) + '.pkl')

        # # do not re-generate data
        # print 'loading data for contour at of val:', of_val
        # branch = np.load('./data/b4' + str(i) + '.pkl')

        label = None
        # only label once
        if i == 0:
            label = r'$c(\theta) = 10^{-4}$'
        ax.plot(branch[:,0], branch[:,1], c='r', label=label, lw=lw)


    ax.set_xlabel(r'$1/\epsilon$')
    ax.set_ylabel(r'$y_0$')
    ax.set_ylim((-6.5, 5.5))
    ax.set_xlim((0, 130))
    ax.legend(fontsize=40, loc=4)
    fig.subplots_adjust(bottom=0.2)
    plt.show()

    # # for investigative purposes
    # sp_system.change_of_val(0.0)
    # npts = 100
    # y0_vals = np.linspace(-5,5,npts)
    # inveps_vals = np.linspace(1, 150, npts) # np.logspace(-5, 0, npts)
    # ofs = np.empty((npts, npts))
    # for i, y0 in enumerate(y0_vals):#uf.parallelize_iterable(y0_vals, rank, nprocs):
    #     for j, eps in enumerate(inveps_vals):
    #         try:
    #             ofs[i,j] = sp_system.get_f_val(eps, y0)
    #         except CustomErrors.IntegrationError:
    #             print 'no'
    #             ofs[i,j] = 1e-14
    #             continue
    # inveps_vals, y0_vals = np.meshgrid(inveps_vals, y0_vals)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.set_xscale('log')
    # c = ax.scatter(inveps_vals, y0_vals, c=np.log10(ofs))
    # fig.colorbar(c)
    # plt.show()
            
def sing_pert_contours_fig2():
    """Plots contours of system used in 'sing_pert_data_space' in y0 and epsilon using arclength continuation"""
    x0 = 1
    lam = 1
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.array((0.01, 0.1, 0.5))
    # # working param set
    # y0_true = 5
    # eps_true = 0.5
    # of_val = 0.1
    # ds = 1e-3
    # nsteps = 2000
    # set up base system used by all of_vals
    y0_true = 5
    epsinv_true = 3 # eps_true = 0.33
    of_val = 0.1
    sp_system = Inverse_Singularly_Perturbed_System(epsinv_true, y0_true, of_val, times)
    # continue along y0
    psa_solver = PSA(sp_system.get_f_val, sp_system.get_Df_val)

    # fig to which every branch will be added
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lw = 2.0

    nof_vals = 4
    of_vals = np.logspace(-4, -1, nof_vals)
    dss = [3e-6, 3e-5, 3e-4, 3e-3]
    colornorm = colors.Normalize(vmin=0, vmax=nof_vals - 1) # colors.Normalize(vmin=np.min(embeddings), vmax=np.max(embeddings))
    colormap = cm.ScalarMappable(norm=colornorm, cmap='viridis_r')

    nsteps = 2000
    y0_guess = 5
    epsinv_guess = 2
    for i, of_val in enumerate([of_vals[2]]):
        i = 2
        # sp_system.change_of_val(of_val)
        # # start solver three times at different points to flesh out the complete contour
        # # (not stable enough to make it around the full ellipse in a computationally feasible time)
        # ds = dss[i]
        # try:
        #     branch = psa_solver.find_branch(np.array((epsinv_guess,)), y0_guess, ds, nsteps, progress_bar=True, abstol=1e-9, reltol=1e-9)
        # except CustomErrors.PSAError as e:
        #     print e.msg
        #     exit()
        # print 'found ' + str(100.0*branch.shape[0]/nsteps) + '% of pts on the level set'
        # branch.dump('./data/large-eps-b' + str(i) + '.pkl')

        # do not re-generate data
        print 'loading data for contour at of val:', of_val
        branch = np.load('./data/large-eps-b' + str(i) + '.pkl')

        label = None
        # only label once
        label = r'$c(\theta) = 10^{-' + str(4-i) + '}$'
        ax.plot(branch[:,0], branch[:,1], c=colormap.to_rgba(i), label=label, lw=lw)


    ax.set_xlabel(r'$1/\epsilon$')
    ax.set_ylabel(r'$y_0$')
    # ax.set_ylim((-6.5, 5.5))
    # ax.set_xlim((0, 130))
    ax.legend(fontsize=40, loc=4)
    fig.subplots_adjust(bottom=0.2)
    plt.show()


    # # for investigative purposes
    # sp_system.change_of_val(0.0)
    # npts = 30
    # y0_vals = np.linspace(0,10,npts)
    # inveps_vals = np.linspace(1, 10, npts) # np.logspace(-5, 0, npts)
    # ofs = np.empty((npts, npts))
    # for i, y0 in enumerate(y0_vals):#uf.parallelize_iterable(y0_vals, rank, nprocs):
    #     for j, eps in enumerate(inveps_vals):
    #         try:
    #             ofs[i,j] = sp_system.get_f_val(eps, y0)
    #         except CustomErrors.IntegrationError:
    #             ofs[i,j] = 1e-14
    #             continue
    # inveps_vals, y0_vals = np.meshgrid(inveps_vals, y0_vals)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.set_xscale('log')
    # c = ax.scatter(inveps_vals, y0_vals, c=np.log10(ofs))
    # fig.colorbar(c)
    # plt.show()


def sing_pert_data_space_fig():
    """Plots three dimensional data space of the classic singularly perturbed ODE system: x' = -lambda*x, y' = -y/epsilon. Probably will end up plotting {y(t1), y(t2), y(t3)} though may also plot norms {|| (x(t1), y(t1)) ||, ... } though this doesn't represent data space in the traditional sense. In particular we're interested in observing the transition from 2 to 1 to 0 dimensional parameter -> data space mappings, i.e. 2 stiff to 1 stiff to 0 stiff parameters."""
    times = np.array((0.01, 0.1, 0.5))
    npts = 40
    lam = 1.0
    x0s = np.ones((npts,2))
    x0s[:,1] = np.linspace(0,10,npts) # x0[0] = 1, x0[1] \in [0,10]
    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)
    cmap = colormaps.viridis

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # start with larger epsilon values (see curvature)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    epss = np.logspace(-2, 0, npts)

    xt1 = np.empty((npts,npts))
    xt2 = np.empty((npts,npts))
    xt3 = np.empty((npts,npts))
    epss_copy = np.empty((npts,npts))
    y0s = np.empty((npts,npts))
    for i, x0 in enumerate(x0s):
        for j, eps in enumerate(epss):
            f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
            integrator = Integration.Integrator(f)
            xt1[i,j], xt2[i,j], xt3[i,j] = integrator.integrate(x0, times)[:,1]
            epss_copy[i,j] = eps
            y0s[i,j] = x0[1]
            
    # save the data
    xt1.dump('./data/sing-pert-x1-large.pkl')
    xt2.dump('./data/sing-pert-x2-large.pkl')
    xt3.dump('./data/sing-pert-x3-large.pkl')
    epss_copy.dump('./data/sing-pert-eps-copy-large.pkl')
    y0s.dump('./data/sing-pert-y0s-large.pkl')

    # # load the data
    # xt1 = np.load('./data/sing-pert-x1-large.pkl')
    # xt2 = np.load('./data/sing-pert-x2-large.pkl')
    # xt3 = np.load('./data/sing-pert-x3-large.pkl')
    # epss_copy = np.load('./data/sing-pert-eps-copy-large.pkl')
    # y0s = np.load('./data/sing-pert-y0s-large.pkl')
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # print 'loading previously generated data for plots'
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # # plot a few trajectories to visualize behavior of ODEs
    # eps_tests = np.logspace(-1, 0, 5)
    # fig_pp = plt.figure()
    # ax_pp = fig_pp.add_subplot(111)
    # fig_y = plt.figure()
    # ax_y = fig_y.add_subplot(111)
    # for i in range(5):
    #     eps = eps_tests[i]
    #     x0 = np.array((1,5))
    #     f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
    #     integrator = Integration.Integrator(f)
    #     ts = np.linspace(0.0001, 1, 1000)
    #     Xs = integrator.integrate(x0, ts)
    #     eps_str = '%1.2f' % eps
    #     ax_pp.plot(Xs[:,0], Xs[:,1], label=r'$\epsilon=$' + eps_str)
    #     ax_y.plot(ts, Xs[:,1], label=r'$\epsilon=$' + eps_str)
    # ax_y.set_xlabel(r'$t$')
    # ax_y.set_ylabel(r'$y$')
    # ax_pp.set_xlabel(r'$x$')
    # ax_pp.set_ylabel(r'$y$')
    # ax_pp.legend(loc=2)
    # ax_y.legend(loc=1)
    # plt.show()

    # # # plot points in 2d parameter space
    # x0_grid, epss_grid = np.meshgrid(x0s, epss)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(epss_grid, x0_grid, s=5)
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\epsilon$')
    # ax.set_ylabel(r'$y(t_0)$')
    # ax.set_xlim((0.09, 1.1))
    # ax.set_ylim((-1, 11))

    # # plot eps and y0 coloring of manifold
    fig1 = plt.figure()
    fig2 = plt.figure()
    # axes
    ax_eps = fig1.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_y0 = fig2.add_subplot(gspec[:,:gsize-1], projection='3d')
    # # make color data
    eps_colors = (1/epss_copy)#/np.max(1/epss_copy)
    eps_colornorm = colors.Normalize(vmin=np.min(eps_colors), vmax=np.max(eps_colors))
    eps_colormap = cm.ScalarMappable(norm=eps_colornorm, cmap=cmap)
    x0_colors = y0s/np.max(y0s)
    # add scatter plots

    ax_eps.plot_surface(xt1, xt2, xt3, facecolors=eps_colormap.to_rgba(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    ax_y0.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    # add unique titles
    # add colorbars
    ax_eps_cb = fig1.add_subplot(gspec[:,gsize-1])
    ax_y0_cb = fig2.add_subplot(gspec[:,gsize-1])
    cb_ticks = np.logspace(np.log10(np.min(1/epss_copy)), np.log10(np.max(1/epss_copy)), 2)
    eps_cb = colorbar.ColorbarBase(ax_eps_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_eps_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)
    y0_cb = colorbar.ColorbarBase(ax_y0_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    ax_y0_cb.text(0.2, 1.05, r'$y(t_0)$', transform=ax_y0_cb.transAxes, fontsize=48)
            
    # # plot a ball in model space
    # find points in ball
    eps_center = 0.33 # center of ball is (eps_center, y0_center) in parameter space
    y0_center = 5.0
    model_space_radius_squared = 0.001
    f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps_center)*(1+10/(1.5-np.sin(x[1])))))), x)
    integrator = Integration.Integrator(f)
    xt1_center, xt2_center, xt3_center = integrator.integrate(np.array((1,y0_center)), times)[:,1]
    # surface_colors = np.zeros((npts,npts,4))
    # # also keep track of which parameters the model manifold points correspond to
    # npts_in_ball = 0
    # eps_in_ball = np.empty(npts*npts)
    # y0_in_ball = np.empty(npts*npts)
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #             if np.power(xt1[i,j] - xt1_center, 2) + np.power(xt2[i,j] - xt2_center, 2) + np.power(xt3[i,j] - xt3_center, 2) < model_space_radius_squared:
    #                 surface_colors[i,j,3] = 0.5 # non-zero alpha
    #                 eps_in_ball[npts_in_ball] = eps
    #                 y0_in_ball[npts_in_ball] = x0[1]
    #                 npts_in_ball = npts_in_ball + 1
    # eps_in_ball = eps_in_ball[:npts_in_ball]
    # y0_in_ball = y0_in_ball[:npts_in_ball]
    # print 'found', npts_in_ball, 'points intersecting ball'

    # plot intersection of full model manifold and ball, colored by x0
    fig = plt.figure()
    ax_ball = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_ball.hold(True)
    # plot surface
    ax_ball.plot_surface(xt1, xt2, xt3, facecolors=eps_colormap.to_rgba(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8, zorder=1)
    # # plot intersection
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=surface_colors, linewidth=0, cstride=1, rstride=1, antialiased=False)
    # plot ball
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    ball_x = np.sqrt(model_space_radius_squared)*np.outer(np.cos(u), np.sin(v))*(np.max(xt1) - np.min(xt1)) + xt1_center
    ball_y = np.sqrt(model_space_radius_squared)*np.outer(np.sin(u), np.sin(v))*(np.max(xt2) - np.min(xt2)) + xt2_center
    ball_z = np.sqrt(model_space_radius_squared)*np.outer(np.ones(np.size(u)), np.cos(v))*(np.max(xt3) - np.min(xt3)) + xt3_center
    ax_ball.plot_surface(ball_x, ball_y, ball_z, color='b', alpha=1.0, linewidth=0, zorder=2)
    # colorbar
    ax_cb = fig.add_subplot(gspec[:,gsize-1])

    ax_ball_cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)

    # # adjust all figure's properties at once as they're all the same
    nticks = 3
    for ax in [ax_eps, ax_y0, ax_ball]:
        formatter = FormatAxis(ax)
        formatter.format('x', xt1, '%1.0f', nticks=nticks)
        formatter.format('y', xt2, '%1.3f', nticks=nticks)
        formatter.format('z', xt3, '%1.3f', nticks=nticks)
        # # axis limits
        # ax.set_xlim((0.99*np.min(xt1), 1.01*np.max(xt1)))
        # ax.set_ylim((0.99*np.min(xt2), 1.01*np.max(xt2)))
        # ax.set_zlim((0.99*np.min(xt3), 1.01*np.max(xt3)))
        # axis labels
        # when have ticks:
        # ax.set_xlabel('\n\n' + r'$y(t_1)$')
        # ax.set_ylabel('\n\n\n' + r'$y(t_2)$')
        # ax.set_zlabel('\n\n\n' + r'$y(t_3)$')
        # when no have ticks:
        ax.set_xlabel('\n' + r'$y(t_1)$')
        ax.set_ylabel('\n' + r'$y(t_2)$')
        ax.set_zlabel('\n' + r'$y(t_3)$')
        # white out the bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # orient axes correctly
        ax.invert_zaxis()
        ax.invert_yaxis()
        # hide (technically incorrect) tick labels
        ax.tick_params(labelsize=0)
        # # hide grid
        # ax.grid(False)
        # # hide tick labels
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # move labels to avoid overlap with numbers
        # ax.xaxis._axinfo['label']['space_factor'] = 2.2
        # ax.yaxis._axinfo['label']['space_factor'] = 2.2
        # ax.zaxis._axinfo['label']['space_factor'] = 2.2

    # # change ax_ball's limits separately to ensure no stretching of the sphere occurs
    # model_space_radius = np.sqrt(model_space_radius_squared)
    # scale = 50
    # ax_ball.set_xlim((xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius))
    # ax_ball.set_ylim((xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius))
    # ax_ball.set_zlim((xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius))
    # formatter = FormatAxis(ax_ball)
    # formatter.format('x', np.linspace(xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius, 10), '%1.0f', nticks=3)
    # formatter.format('y', np.linspace(xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    # formatter.format('z', np.linspace(xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    # ax_ball.set_xlabel('\n\n' + r'$y(t_1)$')
    # ax_ball.set_ylabel('\n\n\n' + r'$y(t_2)$')
    # ax_ball.set_zlabel('\n\n\n' + r'$y(t_3)$')
    # # white out the bg
    # ax_ball.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # plt.show()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # repeat with small epsilon values, ball encompasses manifold corner
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    epss = np.logspace(-2, -1.5, npts)

    # xt1 = np.empty((npts,npts))
    # xt2 = np.empty((npts,npts))
    # xt3 = np.empty((npts,npts))
    # epss_copy = np.empty((npts,npts))
    # y0s = np.empty((npts,npts))
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #         f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
    #         integrator = Integration.Integrator(f)
    #         xt1[i,j], xt2[i,j], xt3[i,j] = integrator.integrate(x0, times)[:,1]
    #         epss_copy[i,j] = eps
    #         y0s[i,j] = x0[1]
            
    # # save the data
    # xt1.dump('./data/sing-pert-x1-small.pkl')
    # xt2.dump('./data/sing-pert-x2-small.pkl')
    # xt3.dump('./data/sing-pert-x3-small.pkl')
    # epss_copy.dump('./data/sing-pert-eps-copy-small.pkl')
    # y0s.dump('./data/sing-pert-y0s-small.pkl')

    # load the data
    xt1 = np.load('./data/sing-pert-x1-small.pkl')
    xt2 = np.load('./data/sing-pert-x2-small.pkl')
    xt3 = np.load('./data/sing-pert-x3-small.pkl')
    epss_copy = np.load('./data/sing-pert-eps-copy-small.pkl')
    y0s = np.load('./data/sing-pert-y0s-small.pkl')
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print 'loading previously generated data for plots'
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # # plot a few trajectories to visualize behavior of ODEs
    # eps_tests = np.logspace(-1, 0, 5)
    # fig_pp = plt.figure()
    # ax_pp = fig_pp.add_subplot(111)
    # fig_y = plt.figure()
    # ax_y = fig_y.add_subplot(111)
    # for i in range(5):
    #     eps = eps_tests[i]
    #     x0 = np.array((1,5))
    #     f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
    #     integrator = Integration.Integrator(f)
    #     ts = np.linspace(0.0001, 1, 1000)
    #     Xs = integrator.integrate(x0, ts)
    #     eps_str = '%1.2f' % eps
    #     ax_pp.plot(Xs[:,0], Xs[:,1], label=r'$\epsilon=$' + eps_str)
    #     ax_y.plot(ts, Xs[:,1], label=r'$\epsilon=$' + eps_str)
    # ax_y.set_xlabel(r'$t$')
    # ax_y.set_ylabel(r'$y$')
    # ax_pp.set_xlabel(r'$x$')
    # ax_pp.set_ylabel(r'$y$')
    # ax_pp.legend(loc=2)
    # ax_y.legend(loc=1)
    # plt.show()

    # # # plot points in 2d parameter space
    # x0_grid, epss_grid = np.meshgrid(x0s, epss)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(epss_grid, x0_grid, s=5)
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\epsilon$')
    # ax.set_ylabel(r'$y(t_0)$')
    # ax.set_xlim((0.09, 1.1))
    # ax.set_ylim((-1, 11))

    # # plot eps and y0 coloring of manifold
    fig1 = plt.figure()
    fig2 = plt.figure()
    # axes
    ax_eps = fig1.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_y0 = fig2.add_subplot(gspec[:,:gsize-1], projection='3d')
    # # make color data
    eps_colors = (1/epss_copy)/np.max(1/epss_copy)
    x0_colors = y0s/np.max(y0s)
    # add scatter plots

    ax_eps.plot_surface(xt1, xt2, xt3, facecolors=eps_colormap.to_rgba(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    ax_y0.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    # add unique titles
    # add colorbars
    ax_eps_cb = fig1.add_subplot(gspec[:,gsize-1])
    ax_y0_cb = fig2.add_subplot(gspec[:,gsize-1])
    eps_cb = colorbar.ColorbarBase(ax_eps_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_eps_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)
    y0_cb = colorbar.ColorbarBase(ax_y0_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    ax_y0_cb.text(0.2, 1.05, r'$y(t_0)$', transform=ax_y0_cb.transAxes, fontsize=48)
            
    # # plot a ball in model space
    # find points in ball
    eps_center = 0.01 # center of ball is (eps_center, y0_center) in parameter space
    y0_center = 5.0
    model_space_radius_squared = 0.001
    f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps_center)*(1+10/(1.5-np.sin(x[1])))))), x)
    integrator = Integration.Integrator(f)
    xt1_center, xt2_center, xt3_center = integrator.integrate(np.array((1,y0_center)), times)[:,1]
    # surface_colors = np.zeros((npts,npts,4))
    # # also keep track of which parameters the model manifold points correspond to
    # npts_in_ball = 0
    # eps_in_ball = np.empty(npts*npts)
    # y0_in_ball = np.empty(npts*npts)
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #             if np.power(xt1[i,j] - xt1_center, 2) + np.power(xt2[i,j] - xt2_center, 2) + np.power(xt3[i,j] - xt3_center, 2) < model_space_radius_squared:
    #                 surface_colors[i,j,3] = 0.5 # non-zero alpha
    #                 eps_in_ball[npts_in_ball] = eps
    #                 y0_in_ball[npts_in_ball] = x0[1]
    #                 npts_in_ball = npts_in_ball + 1
    # eps_in_ball = eps_in_ball[:npts_in_ball]
    # y0_in_ball = y0_in_ball[:npts_in_ball]
    # print 'found', npts_in_ball, 'points intersecting ball'

    # plot intersection of full model manifold and ball, colored by x0
    fig = plt.figure()
    ax_ball = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_ball.hold(True)
    # plot surface
    ax_ball.plot_surface(xt1, xt2, xt3, facecolors=cmap(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=1.0)
    # # plot intersection
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=surface_colors, linewidth=0, cstride=1, rstride=1, antialiased=False)
    # plot ball
    # scale x, y and z components by axis lengths
    u = np.linspace(0, 2*np.pi, 500)
    v = np.linspace(0, np.pi, 500)
    ball_x = np.sqrt(model_space_radius_squared)*np.outer(np.cos(u), np.sin(v))*(np.max(xt1) - np.min(xt1)) + xt1_center
    ball_y = np.sqrt(model_space_radius_squared)*np.outer(np.sin(u), np.sin(v))*(np.max(xt2) - np.min(xt2)) + xt2_center
    ball_z = np.sqrt(model_space_radius_squared)*np.outer(np.ones(np.size(u)), np.cos(v))*(np.max(xt3) - np.min(xt3)) + xt3_center
    ax_ball.plot_surface(ball_x, ball_y, ball_z, color='b', alpha=1.0, linewidth=0)
    # colorbar
    ax_cb = fig.add_subplot(gspec[:,gsize-1])

    ax_ball_cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)

    # # adjust all figure's properties at once as they're all the same
    nticks = 3
    for ax in [ax_eps, ax_y0, ax_ball]:
        formatter = FormatAxis(ax)
        formatter.format('x', xt1, '%1.0f', nticks=nticks)
        formatter.format('y', xt2, '%1.3f', nticks=nticks)
        formatter.format('z', xt3, '%1.3f', nticks=nticks)
        # # axis limits
        # ax.set_xlim((0.99*np.min(xt1), 1.01*np.max(xt1)))
        # ax.set_ylim((0.99*np.min(xt2), 1.01*np.max(xt2)))
        # ax.set_zlim((0.99*np.min(xt3), 1.01*np.max(xt3)))
        # when have ticks:
        # ax.set_xlabel('\n\n' + r'$y(t_1)$')
        # ax.set_ylabel('\n\n\n' + r'$y(t_2)$')
        # ax.set_zlabel('\n\n\n' + r'$y(t_3)$')
        # when no have ticks:
        ax.set_xlabel('\n' + r'$y(t_1)$')
        ax.set_ylabel('\n' + r'$y(t_2)$')
        ax.set_zlabel('\n' + r'$y(t_3)$')
        # white out the bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.invert_zaxis()
        ax.invert_yaxis()
        ax.tick_params(labelsize=0)
        # # hide grid
        # ax.grid(False)
        # # hide tick labels
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # move labels to avoid overlap with numbers
        # ax.xaxis._axinfo['label']['space_factor'] = 2.2
        # ax.yaxis._axinfo['label']['space_factor'] = 2.2
        # ax.zaxis._axinfo['label']['space_factor'] = 2.2

    # change ax_ball's limits separately to ensure no stretching of the sphere occurs
    # model_space_radius = np.sqrt(model_space_radius_squared)
    # ax_ball.set_xlim((xt1_center - 1.1*model_space_radius, xt1_center + 1.1*model_space_radius))
    # ax_ball.set_ylim((xt2_center - 1.1*model_space_radius, xt2_center + 1.1*model_space_radius))
    # ax_ball.set_zlim((xt3_center - 1.1*model_space_radius, xt3_center + 1.1*model_space_radius))
    # ax_ball.set_xlabel('\n\n' + r'$y(t_1)$')
    # ax_ball.set_ylabel('\n\n\n' + r'$y(t_2)$')
    # ax_ball.set_zlabel('\n\n\n' + r'$y(t_3)$')
    # # white out the bg
    # ax_ball.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    plt.show()
    
def sing_pert_data_space_fig_test2_gendata():
    """Plots three dimensional data space of the classic singularly perturbed ODE system: x' = -lambda*x, y' = -y/epsilon. Probably will end up plotting {y(t1), y(t2), y(t3)} though may also plot norms {|| (x(t1), y(t1)) ||, ... } though this doesn't represent data space in the traditional sense. In particular we're interested in observing the transition from 2 to 1 to 0 dimensional parameter -> data space mappings, i.e. 2 stiff to 1 stiff to 0 stiff parameters."""
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    times = np.array((0.01, 0.1, 0.5))
    npts = 1000
    lam = 1.0
    x0s = np.ones((npts,2))
    x0s[:,1] = np.linspace(-5,5,npts) # x0[0] = 1, x0[1] \in [0,10]
    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)
    cmap = colormaps.viridis

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # start with larger epsilon values (see curvature)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    epss = np.logspace(-2, 0, npts)

    xt1 = np.zeros((npts,npts))
    xt2 = np.zeros((npts,npts))
    xt3 = np.zeros((npts,npts))
    epss_copy = np.zeros((npts,npts))
    y0s = np.zeros((npts,npts))
    if rank == 0:
        xt1o = np.zeros((npts,npts))
        xt2o = np.zeros((npts,npts))
        xt3o = np.zeros((npts,npts))
        epss_copyo = np.zeros((npts,npts))
        y0so = np.zeros((npts,npts))
    else:
        xt1o = None
        xt2o = None
        xt3o = None
        epss_copyo = None
        y0so = None

    for i, x0 in uf.parallelize_iterable(zip(range(npts), x0s), rank, nprocs):
        for j, eps in enumerate(epss):
            f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
            integrator = Integration.Integrator(f)
            xt1[i,j], xt2[i,j], xt3[i,j] = integrator.integrate(x0, times)[:,1]
            epss_copy[i,j] = eps
            y0s[i,j] = x0[1]
            
    # gather the data
    comm.Reduce([xt1, MPI.DOUBLE], [xt1o, MPI.DOUBLE], op = MPI.SUM, root = 0)
    comm.Reduce([xt2, MPI.DOUBLE], [xt2o, MPI.DOUBLE], op = MPI.SUM, root = 0)
    comm.Reduce([xt3, MPI.DOUBLE], [xt3o, MPI.DOUBLE], op = MPI.SUM, root = 0)
    comm.Reduce([epss_copy, MPI.DOUBLE], [epss_copyo, MPI.DOUBLE], op = MPI.SUM, root = 0)
    comm.Reduce([y0s, MPI.DOUBLE], [y0so, MPI.DOUBLE], op = MPI.SUM, root = 0)

    # save the data
    if rank == 0:
        xt1o.dump('./data/sing-pert-x1-large.pkl')
        xt2o.dump('./data/sing-pert-x2-large.pkl')
        xt3o.dump('./data/sing-pert-x3-large.pkl')
        epss_copyo.dump('./data/sing-pert-eps-copy-large.pkl')
        y0so.dump('./data/sing-pert-y0s-large.pkl')
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        print 'saved the data'
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

def sing_pert_data_space_fig_test2_plotdata():

    # # load the data
    # xt1 = np.load('./data/sing-pert-x1-large.pkl')
    # xt2 = np.load('./data/sing-pert-x2-large.pkl')
    # xt3 = np.load('./data/sing-pert-x3-large.pkl')
    # epss_copy = np.load('./data/sing-pert-eps-copy-large.pkl')
    # y0s = np.load('./data/sing-pert-y0s-large.pkl')
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # print 'loading previously generated data for plots'
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # # plot a few trajectories to visualize behavior of ODEs
    # eps_tests = np.logspace(-1, 0, 5)
    # fig_pp = plt.figure()
    # ax_pp = fig_pp.add_subplot(111)
    # fig_y = plt.figure()
    # ax_y = fig_y.add_subplot(111)
    # for i in range(5):
    #     eps = eps_tests[i]
    #     x0 = np.array((1,5))
    #     f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
    #     integrator = Integration.Integrator(f)
    #     ts = np.linspace(0.0001, 1, 1000)
    #     Xs = integrator.integrate(x0, ts)
    #     eps_str = '%1.2f' % eps
    #     ax_pp.plot(Xs[:,0], Xs[:,1], label=r'$\epsilon=$' + eps_str)
    #     ax_y.plot(ts, Xs[:,1], label=r'$\epsilon=$' + eps_str)
    # ax_y.set_xlabel(r'$t$')
    # ax_y.set_ylabel(r'$y$')
    # ax_pp.set_xlabel(r'$x$')
    # ax_pp.set_ylabel(r'$y$')
    # ax_pp.legend(loc=2)
    # ax_y.legend(loc=1)
    # plt.show()

    # # # plot points in 2d parameter space
    # x0_grid, epss_grid = np.meshgrid(x0s, epss)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(epss_grid, x0_grid, s=5)
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\epsilon$')
    # ax.set_ylabel(r'$y(t_0)$')
    # ax.set_xlim((0.09, 1.1))
    # ax.set_ylim((-1, 11))

    # # plot eps and y0 coloring of manifold
    fig1 = plt.figure()
    fig2 = plt.figure()
    # axes
    ax_eps = fig1.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_y0 = fig2.add_subplot(gspec[:,:gsize-1], projection='3d')
    # # make color data
    eps_colors = (1/epss_copy)#/np.max(1/epss_copy)
    eps_colornorm = colors.Normalize(vmin=np.min(eps_colors), vmax=np.max(eps_colors))
    eps_colormap = cm.ScalarMappable(norm=eps_colornorm, cmap=cmap)
    x0_colors = y0s/np.max(y0s)
    # add scatter plots

    ax_eps.plot_surface(xt1, xt2, xt3, facecolors=eps_colormap.to_rgba(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    ax_y0.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    # add unique titles
    # add colorbars
    ax_eps_cb = fig1.add_subplot(gspec[:,gsize-1])
    ax_y0_cb = fig2.add_subplot(gspec[:,gsize-1])
    cb_ticks = np.logspace(np.log10(np.min(1/epss_copy)), np.log10(np.max(1/epss_copy)), 2)
    eps_cb = colorbar.ColorbarBase(ax_eps_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_eps_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)
    y0_cb = colorbar.ColorbarBase(ax_y0_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    ax_y0_cb.text(0.2, 1.05, r'$y(t_0)$', transform=ax_y0_cb.transAxes, fontsize=48)
            
    # # plot a ball in model space
    # find points in ball
    eps_center = 0.33 # center of ball is (eps_center, y0_center) in parameter space
    y0_center = 5.0
    model_space_radius_squared = 0.001
    f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps_center)*(1+10/(1.5-np.sin(x[1])))))), x)
    integrator = Integration.Integrator(f)
    xt1_center, xt2_center, xt3_center = integrator.integrate(np.array((1,y0_center)), times)[:,1]
    # surface_colors = np.zeros((npts,npts,4))
    # # also keep track of which parameters the model manifold points correspond to
    # npts_in_ball = 0
    # eps_in_ball = np.empty(npts*npts)
    # y0_in_ball = np.empty(npts*npts)
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #             if np.power(xt1[i,j] - xt1_center, 2) + np.power(xt2[i,j] - xt2_center, 2) + np.power(xt3[i,j] - xt3_center, 2) < model_space_radius_squared:
    #                 surface_colors[i,j,3] = 0.5 # non-zero alpha
    #                 eps_in_ball[npts_in_ball] = eps
    #                 y0_in_ball[npts_in_ball] = x0[1]
    #                 npts_in_ball = npts_in_ball + 1
    # eps_in_ball = eps_in_ball[:npts_in_ball]
    # y0_in_ball = y0_in_ball[:npts_in_ball]
    # print 'found', npts_in_ball, 'points intersecting ball'

    # plot intersection of full model manifold and ball, colored by x0
    fig = plt.figure()
    ax_ball = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_ball.hold(True)
    # plot surface
    ax_ball.plot_surface(xt1, xt2, xt3, facecolors=eps_colormap.to_rgba(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8, zorder=1)
    # # plot intersection
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=surface_colors, linewidth=0, cstride=1, rstride=1, antialiased=False)
    # plot ball
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    ball_x = np.sqrt(model_space_radius_squared)*np.outer(np.cos(u), np.sin(v))*(np.max(xt1) - np.min(xt1)) + xt1_center
    ball_y = np.sqrt(model_space_radius_squared)*np.outer(np.sin(u), np.sin(v))*(np.max(xt2) - np.min(xt2)) + xt2_center
    ball_z = np.sqrt(model_space_radius_squared)*np.outer(np.ones(np.size(u)), np.cos(v))*(np.max(xt3) - np.min(xt3)) + xt3_center
    ax_ball.plot_surface(ball_x, ball_y, ball_z, color='b', alpha=1.0, linewidth=0, zorder=2)
    # colorbar
    ax_cb = fig.add_subplot(gspec[:,gsize-1])

    ax_ball_cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=eps_colornorm, ticks=cb_ticks)
    ax_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)

    # # adjust all figure's properties at once as they're all the same
    nticks = 3
    for ax in [ax_eps, ax_y0, ax_ball]:
        formatter = FormatAxis(ax)
        formatter.format('x', xt1, '%1.0f', nticks=nticks)
        formatter.format('y', xt2, '%1.3f', nticks=nticks)
        formatter.format('z', xt3, '%1.3f', nticks=nticks)
        # # axis limits
        # ax.set_xlim((0.99*np.min(xt1), 1.01*np.max(xt1)))
        # ax.set_ylim((0.99*np.min(xt2), 1.01*np.max(xt2)))
        # ax.set_zlim((0.99*np.min(xt3), 1.01*np.max(xt3)))
        # axis labels
        # when have ticks:
        # ax.set_xlabel('\n\n' + r'$y(t_1)$')
        # ax.set_ylabel('\n\n\n' + r'$y(t_2)$')
        # ax.set_zlabel('\n\n\n' + r'$y(t_3)$')
        # when no have ticks:
        ax.set_xlabel('\n' + r'$y(t_1)$')
        ax.set_ylabel('\n' + r'$y(t_2)$')
        ax.set_zlabel('\n' + r'$y(t_3)$')
        # white out the bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # orient axes correctly
        ax.invert_zaxis()
        ax.invert_yaxis()
        # hide (technically incorrect) tick labels
        ax.tick_params(labelsize=0)
        # # hide grid
        # ax.grid(False)
        # # hide tick labels
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # move labels to avoid overlap with numbers
        # ax.xaxis._axinfo['label']['space_factor'] = 2.2
        # ax.yaxis._axinfo['label']['space_factor'] = 2.2
        # ax.zaxis._axinfo['label']['space_factor'] = 2.2

    # # change ax_ball's limits separately to ensure no stretching of the sphere occurs
    # model_space_radius = np.sqrt(model_space_radius_squared)
    # scale = 50
    # ax_ball.set_xlim((xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius))
    # ax_ball.set_ylim((xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius))
    # ax_ball.set_zlim((xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius))
    # formatter = FormatAxis(ax_ball)
    # formatter.format('x', np.linspace(xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius, 10), '%1.0f', nticks=3)
    # formatter.format('y', np.linspace(xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    # formatter.format('z', np.linspace(xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    # ax_ball.set_xlabel('\n\n' + r'$y(t_1)$')
    # ax_ball.set_ylabel('\n\n\n' + r'$y(t_2)$')
    # ax_ball.set_zlabel('\n\n\n' + r'$y(t_3)$')
    # # white out the bg
    # ax_ball.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax_ball.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # plt.show()


def sing_pert_data_space_fig_test():
    """Plots three dimensional data space of the classic singularly perturbed ODE system: x' = -lambda*x, y' = -y/epsilon. Probably will end up plotting {y(t1), y(t2), y(t3)} though may also plot norms {|| (x(t1), y(t1)) ||, ... } though this doesn't represent data space in the traditional sense. In particular we're interested in observing the transition from 2 to 1 to 0 dimensional parameter -> data space mappings, i.e. 2 stiff to 1 stiff to 0 stiff parameters."""
    times = np.array((0.1, 0.15, 0.2))
    npts = 40
    lam = 1.0
    x0s = np.ones((npts,2))
    x0s[:,1] = np.linspace(0,10,npts) # x0[0] = 1, x0[1] \in [0,10]
    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)
    cmap = colormaps.viridis

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # repeat for larger epsilon values (see curvature)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    epss = np.logspace(-1, 0, npts)
    xt1 = np.empty((npts,npts))
    xt2 = np.empty((npts,npts))
    xt3 = np.empty((npts,npts))
    epss_copy = np.empty((npts,npts))
    y0s = np.empty((npts,npts))
    for i, x0 in enumerate(x0s):
        for j, eps in enumerate(epss):
            f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
            integrator = Integration.Integrator(f)
            xt1[i,j], xt2[i,j], xt3[i,j] = integrator.integrate(x0, times)[:,1]
            epss_copy[i,j] = eps
            y0s[i,j] = x0[1]
            
    # save the data
    xt1.dump('./data/sing-pert-x1.pkl')
    xt2.dump('./data/sing-pert-x2.pkl')
    xt3.dump('./data/sing-pert-x3.pkl')
    epss_copy.dump('./data/sing-pert-eps-copy.pkl')
    y0s.dump('./data/sing-pert-y0s.pkl')

    # # load the data
    # xt1 = np.load('./data/sing-pert-x1.pkl')
    # xt2 = np.load('./data/sing-pert-x2.pkl')
    # xt3 = np.load('./data/sing-pert-x3.pkl')
    # epss_copy = np.load('./data/sing-pert-eps-copy.pkl')
    # y0s = np.load('./data/sing-pert-y0s.pkl')
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # print 'loading previously generated data for plots'
    # print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    # # plot a few trajectories to visualize behavior of ODEs
    # eps_tests = np.logspace(-1, 0, 5)
    # fig_pp = plt.figure()
    # ax_pp = fig_pp.add_subplot(111)
    # fig_y = plt.figure()
    # ax_y = fig_y.add_subplot(111)
    # for i in range(5):
    #     eps = eps_tests[i]
    #     x0 = np.array((1,5))
    #     f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps)*(1+10/(1.5-np.sin(x[1])))))), x)
    #     integrator = Integration.Integrator(f)
    #     ts = np.linspace(0.0001, 1, 1000)
    #     Xs = integrator.integrate(x0, ts)
    #     eps_str = '%1.2f' % eps
    #     ax_pp.plot(Xs[:,0], Xs[:,1], label=r'$\epsilon=$' + eps_str)
    #     ax_y.plot(ts, Xs[:,1], label=r'$\epsilon=$' + eps_str)
    # ax_y.set_xlabel(r'$t$')
    # ax_y.set_ylabel(r'$y$')
    # ax_pp.set_xlabel(r'$x$')
    # ax_pp.set_ylabel(r'$y$')
    # ax_pp.legend(loc=2)
    # ax_y.legend(loc=1)
    # plt.show()

    # # # plot points in 2d parameter space
    # x0_grid, epss_grid = np.meshgrid(x0s, epss)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(epss_grid, x0_grid, s=5)
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\epsilon$')
    # ax.set_ylabel(r'$y(t_0)$')
    # ax.set_xlim((0.09, 1.1))
    # ax.set_ylim((-1, 11))

    # # plot eps and y0 coloring of manifold
    fig1 = plt.figure()
    fig2 = plt.figure()
    # axes
    ax_eps = fig1.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_y0 = fig2.add_subplot(gspec[:,:gsize-1], projection='3d')
    # # make color data
    eps_colors = (1/epss_copy)/np.max(1/epss_copy)
    x0_colors = y0s/np.max(y0s)
    # add scatter plots

    ax_eps.plot_surface(xt1, xt2, xt3, facecolors=cmap(eps_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    ax_y0.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8)
    # add unique titles
    # add colorbars
    ax_eps_cb = fig1.add_subplot(gspec[:,gsize-1])
    ax_y0_cb = fig2.add_subplot(gspec[:,gsize-1])
    eps_cb = colorbar.ColorbarBase(ax_eps_cb, cmap=cmap, norm=colors.Normalize(np.min(1/epss_copy), np.max(1/epss_copy)), ticks=np.logspace(np.log10(np.min(1/epss_copy)), np.log10(np.max(1/epss_copy)), 2))
    ax_eps_cb.text(0.2, 1.05, r'$1/\epsilon$', transform=ax_eps_cb.transAxes, fontsize=48)
    y0_cb = colorbar.ColorbarBase(ax_y0_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    ax_y0_cb.text(0.2, 1.05, r'$y(t_0)$', transform=ax_y0_cb.transAxes, fontsize=48)
            
    # # plot a ball in model space
    # find points in ball
    eps_center = 0.33 # center of ball is (eps_center, y0_center) in parameter space
    y0_center = 5.0
    model_space_radius_squared = 0.0001
    f = lambda t, x: np.dot(np.array(((-lam, 1),(1,(-1/eps_center)*(1+10/(1.5-np.sin(x[1])))))), x)
    integrator = Integration.Integrator(f)
    xt1_center, xt2_center, xt3_center = integrator.integrate(np.array((1,y0_center)), times)[:,1]
    # surface_colors = np.zeros((npts,npts,4))
    # # also keep track of which parameters the model manifold points correspond to
    # npts_in_ball = 0
    # eps_in_ball = np.empty(npts*npts)
    # y0_in_ball = np.empty(npts*npts)
    # for i, x0 in enumerate(x0s):
    #     for j, eps in enumerate(epss):
    #             if np.power(xt1[i,j] - xt1_center, 2) + np.power(xt2[i,j] - xt2_center, 2) + np.power(xt3[i,j] - xt3_center, 2) < model_space_radius_squared:
    #                 surface_colors[i,j,3] = 0.5 # non-zero alpha
    #                 eps_in_ball[npts_in_ball] = eps
    #                 y0_in_ball[npts_in_ball] = x0[1]
    #                 npts_in_ball = npts_in_ball + 1
    # eps_in_ball = eps_in_ball[:npts_in_ball]
    # y0_in_ball = y0_in_ball[:npts_in_ball]
    # print 'found', npts_in_ball, 'points intersecting ball'

    # plot intersection of full model manifold and ball, colored by x0
    fig = plt.figure()
    ax_ball = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    ax_ball.hold(True)
    # plot surface
    ax_ball.plot_surface(xt1, xt2, xt3, facecolors=cmap(x0_colors), edgecolors=np.array((0,0,0,0)), linewidth=0, shade=False, cstride=1, rstride=1, alpha=0.8, zorder=1)
    # # plot intersection
    # ax_ball.plot_surface(xt1, xt2, xt3, facecolors=surface_colors, linewidth=0, cstride=1, rstride=1, antialiased=False)
    # plot ball
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ball_x = np.sqrt(model_space_radius_squared)*np.outer(np.cos(u), np.sin(v)) + xt1_center
    ball_y = np.sqrt(model_space_radius_squared)*np.outer(np.sin(u), np.sin(v)) + xt2_center
    ball_z = np.sqrt(model_space_radius_squared)*np.outer(np.ones(np.size(u)), np.cos(v)) + xt3_center
    ax_ball.plot_surface(ball_x, ball_y, ball_z, color='b', alpha=1.0, linewidth=0, zorder=2)
    # colorbar
    ax_cb = fig.add_subplot(gspec[:,gsize-1])
    y0_cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colors.Normalize(np.min(y0s), np.max(y0s)), ticks=np.linspace(np.min(x0s), np.max(y0s), 5))
    ax_cb.text(0.2, 1.05, r'$y(t_0)$', transform=ax_cb.transAxes, fontsize=48)


    # # adjust all figure's properties at once as they're all the same
    for ax in [ax_eps, ax_y0]:
        formatter = FormatAxis(ax)
        formatter.format('x', xt1, '%1.0f', nticks=3)
        formatter.format('y', xt2, '%1.3f', nticks=3)
        formatter.format('z', xt3, '%1.3f', nticks=3)
        # # axis limits
        # ax.set_xlim((0.99*np.min(xt1), 1.01*np.max(xt1)))
        # ax.set_ylim((0.99*np.min(xt2), 1.01*np.max(xt2)))
        # ax.set_zlim((0.99*np.min(xt3), 1.01*np.max(xt3)))
        # axis labels
        ax.set_xlabel('\n\n' + r'$y(t_1)$')
        ax.set_ylabel('\n\n\n' + r'$y(t_2)$')
        ax.set_zlabel('\n\n\n' + r'$y(t_3)$')
        # white out the bg
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # hide grid
        # ax.grid(False)
        # # hide tick labels
        # plt.tick_params(axis='both', which='major', labelsize=0)
        # move labels to avoid overlap with numbers
        # ax.xaxis._axinfo['label']['space_factor'] = 2.2
        # ax.yaxis._axinfo['label']['space_factor'] = 2.2
        # ax.zaxis._axinfo['label']['space_factor'] = 2.2

    # change ax_ball's limits separately to ensure no stretching of the sphere occurs
    model_space_radius = np.sqrt(model_space_radius_squared)
    scale = 50
    ax_ball.set_xlim((xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius))
    ax_ball.set_ylim((xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius))
    ax_ball.set_zlim((xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius))
    formatter = FormatAxis(ax_ball)
    formatter.format('x', np.linspace(xt1_center - scale*model_space_radius, xt1_center + scale*model_space_radius, 10), '%1.0f', nticks=3)
    formatter.format('y', np.linspace(xt2_center - scale*model_space_radius, xt2_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    formatter.format('z', np.linspace(xt3_center - scale*model_space_radius, xt3_center + scale*model_space_radius, 10), '%1.3f', nticks=3)
    ax_ball.set_xlabel('\n\n' + r'$y(t_1)$')
    ax_ball.set_ylabel('\n\n\n' + r'$y(t_2)$')
    ax_ball.set_zlabel('\n\n\n' + r'$y(t_3)$')
    # white out the bg
    ax_ball.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_ball.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_ball.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    plt.show()

def linear_sing_pert_model_manifold_fig():
    """Plotting model manifold of y' = -1/eps * y"""

    cmap ='viridis'

    gray_color = np.array((0.05, 0.05, 0.05, 0.1))
    
    gsize = 40
    gspec = gs.GridSpec(gsize, gsize)

    t1, t2, t3 = (1.,2.,6.)# np.linspace(1,3,3)
    # constants for parameter space plot, also needed when making model manifold
    y0max = 2
    logeps_min = -3
    logeps_max = 3

    npts = 200
    xss = np.linspace(0,1,npts)
    yss = np.linspace(0,1,npts)
    xg, yg = np.meshgrid(xss,yss)
    xkept = np.empty(npts*npts)
    ykept = np.empty(npts*npts)
    count = 0
    origin = None
    for i in range(npts):
        for j in range(npts):
            if yg[i,j] <= xg[i,j] and xg[i,j] <= y0max*np.exp(-t1*np.log(y0max/yg[i,j])/t2):
                # keep track of which point is the origin for subsequent settings of y0, eps and z to 0 or inf
                if xg[i,j] == 0 and yg[i,j] == 0:
                    origin = count
                xkept[count] = xg[i,j]
                ykept[count] = yg[i,j]
                count = count + 1
    xkept = xkept[:count]
    ykept = ykept[:count]

    z = np.power(xkept, (t2 - t3)/(t2 - t1))*np.power(ykept, (t3 - t1)/(t2 - t1))
    z[origin] = 0 # z(0,0) = 0

    epss = (t2 - t1)/np.log(xkept/ykept)
    epss[np.isnan(epss)] = np.inf # eps(0,0,0) = inf
    epss_transformed = np.tanh(epss)

    y0s = xkept*np.power(xkept/ykept, t1/(t2 - t1))
    y0s[origin] = 0 # y0(origin) = 0
    y0s[np.isnan(y0s)] = np.inf
    y0s_transformed = np.tanh(y0s)

    xyz = np.array((xkept, ykept, z)).T

    # y0s_colornorm = colors.Normalize(vmin=np.min(y0s_transformed), vmax=np.max(y0s_transformed))
    y0s_colornorm = colors.Normalize(vmin=0, vmax=y0max)
    y0s_colormap = cm.ScalarMappable(norm=y0s_colornorm, cmap=cmap)

    tris = tri.Triangulation(xkept, ykept).triangles
    triangle_vertices = np.array([np.array((xyz[T[0]], xyz[T[1]], xyz[T[2]])) for T in tris]) # plot in k1, k-1, k2
    # create polygon collections that will be plotted (seems impossible to completely erase edges)

    # # first attempt to define the two-variable colormap (y0 -> color, eps -> alpha)
    # face_colors = y0s_colormap.to_rgba(calc_tri_avg(tris, y0s_transformed))
    # face_colors = colors.rgb_to_hsv(face_colors[:,:3])

    # saturation_values = calc_tri_avg(tris, epss_transformed)
    # saturation_values = 0.3 + 0.6*saturation_values/np.max(saturation_values)
    # face_colors[:,1] = saturation_values
    # face_colors[:,2] = 1
    # face_colors = colors.hsv_to_rgb(face_colors)
    # coll_y0 = Poly3DCollection(triangle_vertices, linewidths=0, alpha=0.75)
    # coll_y0.set_facecolor(face_colors)
    # coll_y0.set_edgecolor(face_colors)
    # # end attempt

    face_colors = y0s_colormap.to_rgba(calc_tri_avg(tris, y0s))

    coll_y0 = Poly3DCollection(triangle_vertices, linewidths=0)
    coll_y0.set_facecolor(face_colors)
    coll_y0.set_edgecolor(face_colors)
    # coll_eps = Poly3DCollection(triangle_vertices, facecolors=eps_colormap.to_rgba(calc_tri_avg(tris, epss_transformed)), linewidths=0, edgecolors='w')

    # add greyed-area that doesn't correspond to region in parameter space
    xkept = np.empty(npts*npts)
    ykept = np.empty(npts*npts)
    count = 0
    origin = None
    arb_tol = 0.00
    for i in range(npts):
        for j in range(npts):
            if yg[i,j] <= xg[i,j] and xg[i,j] >= y0max*np.exp(-t1*np.log(y0max/yg[i,j])/t2) - arb_tol:
                # keep track of which point is the origin for subsequent settings of y0, eps and z to 0 or inf
                if xg[i,j] == 0 and yg[i,j] == 0:
                    origin = count
                xkept[count] = xg[i,j]
                ykept[count] = yg[i,j]
                count = count + 1
    xkept = xkept[:count]
    ykept = ykept[:count]

    # plt.scatter(xkept, ykept)
    # plt.show()

    z = np.power(xkept, (t2 - t3)/(t2 - t1))*np.power(ykept, (t3 - t1)/(t2 - t1))
    z[origin] = 0 # z(0,0) = 0

    xyz = np.array((xkept, ykept, z)).T
    tris = tri.Triangulation(xkept, ykept).triangles
    triangle_vertices = np.array([np.array((xyz[T[0]], xyz[T[1]], xyz[T[2]])) for T in tris]) # plot in k1, k-1, k2

    face_colors_grey = np.ones((tris.shape[0],4))*gray_color
    face_y1s = calc_tri_avg(tris, xkept)
    face_y2s = calc_tri_avg(tris, ykept)
    arb_tol = 0.00
    for i in range(tris.shape[0]):
        if face_y1s[i] < y0max*np.exp(-t1*np.log(y0max/face_y2s[i])/t2) - arb_tol:
            face_colors_grey[i] = 0

    coll_y0_grey = Poly3DCollection(triangle_vertices, linewidths=0)
    coll_y0_grey.set_facecolor(face_colors_grey)
    # coll_y0_grey.set_edgecolor((1,1,1,1))
    # coll_y0_grey.set_edgecolor(gray_color)

    
    # plot both greyed and colored surface
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,gsize/2:], projection='3d')
    ax.add_collection(coll_y0)
    ax.add_collection(coll_y0_grey)

    # ax_cb = fig.add_subplot(gspec[:,gsize-1])
    # cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=y0s_colornorm, ticks=[0,1.,y0max], label=r'$x_0$')
    # cb.ax.set_yticklabels(['0', '1', str(y0max)])

    # add lines of constant epsilon
    epss_samples = np.array((1e-3, 1, 3.5, 10, 1e3)) # hand picked for nice visuals
    # eps_colornorm = colors.Normalize(vmin=np.min(np.log10(epss_samples)), vmax=1.5*np.max(np.log10(epss_samples))) # scale vmax by 1.5 to remove whiter part of cmap from use
    # eps_colormap = cm.ScalarMappable(norm=eps_colornorm, cmap='Oranges_r')
    eps_colors = ['#002255', '#0044aa', '#0066ff', '#5599ff', '#aaccff'] # ['#000000', '#0000cc', '#0099ff', '#cc00ff', '#ff0000']
    lw = 4
    for i, eps in enumerate(epss_samples):
        xss = np.linspace(0,np.min((1, y0max*np.exp(-t1/eps))),npts)
        yss = xss*np.exp((t1 - t2)/eps)
        zss = np.power(xss, (t2 - t3)/(t2 - t1))*np.power(yss, (t3 - t1)/(t2 - t1))
        ax.plot(xss, yss, zss, lw=lw, color=eps_colors[i])#eps_colormap.to_rgba(np.log10(eps)))

    ax.set_xlabel('\n' + r'$f_1$')
    ax.set_ylabel('\n' + r'$f_2$')
    ax.set_zlabel('\n' + r'$f_3$')
    formatter = FormatAxis(ax)
    ax.set_xticks((0,1))
    ax.set_yticks((0,1))
    ax.set_zticks((0,1))
    # formatter.format('x', xkept, '%1.0f', nticks=2)
    # formatter.format('y', ykept, '%1.0f', nticks=2)
    # formatter.format('z', z[::-1], '%1.0f', nticks=2)
    # ax.invert_zaxis()
    ax.invert_yaxis()


    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # START EXPERIMENTAL
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # npts = 100
    # xss = np.linspace(1,,1.5,npts)
    # yss = np.linspace(1,1.5,npts)
    # xg, yg = np.meshgrid(xss,yss)
    # xkept = np.empty(npts*npts)
    # ykept = np.empty(npts*npts)
    # count = 0
    # origin = None
    # for i in range(npts):
    #     for j in range(npts):
    #         if yg[i,j] <= xg[i,j]:
    #             # keep track of which point is the origin for subsequent settings of y0, eps and z to 0 or inf
    #             if xg[i,j] == 0:
    #                 origin = count
    #             xkept[count] = xg[i,j]
    #             ykept[count] = yg[i,j]
    #             count = count + 1
    # xkept = xkept[:count]
    # ykept = ykept[:count]

    # z = np.power(xkept, (t2 - t3)/(t2 - t1))*np.power(ykept, (t3 - t1)/(t2 - t1))
    # z[origin] = 0 # z(0,0) = 0

    # epss = (t2 - t1)/np.log(xkept/ykept)
    # epss[np.isnan(epss)] = np.inf # eps(0,0,0) = inf
    # epss_transformed = np.tanh(epss)

    # y0s = xkept*np.power(xkept/ykept, t1/(t2 - t1))
    # y0s[origin] = 0 # y0(origin) = 0
    # y0s[np.isnan(y0s)] = np.inf
    # y0s_transformed = np.tanh(y0s)

    # xyz = np.array((xkept, ykept, z)).T

    # y0s_colornorm = colors.Normalize(vmin=np.min(y0s_transformed), vmax=np.max(y0s_transformed))
    # y0s_colormap = cm.ScalarMappable(norm=y0s_colornorm, cmap=cmap)

    # tris = tri.Triangulation(xkept, ykept).triangles
    # triangle_vertices = np.array([np.array((xyz[T[0]], xyz[T[1]], xyz[T[2]])) for T in tris]) # plot in k1, k-1, k2
    # # create polygon collections that will be plotted (seems impossible to completely erase edges)
    # coll_y0 = Poly3DCollection(triangle_vertices, facecolors=y0s_colormap.to_rgba(calc_tri_avg(tris, y0s_transformed)), linewidths=0, edgecolors='w')
    # # coll_eps = Poly3DCollection(triangle_vertices, facecolors=eps_colormap.to_rgba(calc_tri_avg(tris, epss_transformed)), linewidths=0, edgecolors='w')
    
    # # plot surface
    # fig = plt.figure()
    # ax = fig.add_subplot(gspec[:,gsize/2:gsize-1], projection='3d')
    # ax.add_collection(coll_y0)

    # ax_cb = fig.add_subplot(gspec[:,gsize-1])
    # cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=y0s_colornorm, ticks=[0,1.], label=r'$x_0$')
    # cb.ax.set_yticklabels([r'$0$', r'$\infty$'])

    # # add lines of constant epsilon
    # epss_samples = np.array((1e-3, 1, 3.5, 10, 1e3)) # hand picked for nice visuals
    # eps_colornorm = colors.Normalize(vmin=np.min(np.log10(epss_samples)), vmax=1.5*np.max(np.log10(epss_samples))) # scale vmax by 1.5 to remove whiter part of cmap from use
    # eps_colormap = cm.ScalarMappable(norm=eps_colornorm, cmap='Oranges_r')
    # eps_colors = ['#000000', '#0000cc', '#0099ff', '#cc00ff', '#ff0000']
    # lw = 3
    # for i, eps in enumerate(epss_samples):
    #     xss = np.linspace(0,1,npts)
    #     yss = xss*np.exp((t1 - t2)/eps)
    #     zss = np.power(xss, (t2 - t3)/(t2 - t1))*np.power(yss, (t3 - t1)/(t2 - t1))
    #     ax.plot(xss, yss, zss, lw=lw, color=eps_colors[i])#eps_colormap.to_rgba(np.log10(eps)))

    # ax.set_xlabel('\n' + r'$f_1$')
    # ax.set_ylabel('\n' + r'$f_2$')
    # ax.set_zlabel('\n' + r'$f_3$')
    # formatter = FormatAxis(ax)
    # formatter.format('x', xkept, '%1.0f', nticks=2)
    # formatter.format('y', ykept, '%1.0f', nticks=2)
    # formatter.format('z', z[::-1], '%1.0f', nticks=2)
    # ax.invert_zaxis()
    # ax.invert_yaxis()

    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # END EXPERIMENTAL
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # plot y0, eps plane
    npts = 1000
    y0s = np.linspace(y0max,0,npts) # reverse ordering to get meshgrid output correctly oriented
    epss = np.logspace(logeps_min, logeps_max, npts)
    epsg, y0g = np.meshgrid(epss, y0s)

    # # (attempt to) set up two-dimensional color grid
    # y0s_colornorm = colors.Normalize(vmin=0, vmax=y0max)
    # y0s_colormap = cm.ScalarMappable(norm=y0s_colornorm, cmap=cmap)
    # color_grid = np.empty((npts, npts, 4))
    # alpha = 0.8
    # for i in range(npts):
    #     for j in range(npts):
    #         custom_color = y0s_colormap.to_rgba(y0g[i,j])
    #         custom_color = colors.rgb_to_hsv(custom_color[:3])
    #         custom_color[1] = 0.25 + 0.75*(np.log10(epsg[i,j]) - logeps_min)/(logeps_max - logeps_min)
    #         custom_color[2] = 1
    #         color_grid[i,j,:3] = colors.hsv_to_rgb(custom_color)
    #         color_grid[i,j,3] = alpha
    # ax.imshow(color_grid, extent=(1e-3,1e3,0,2))
    # # end attempt


    ax = fig.add_subplot(gspec[:,:gsize/2-2])
    lw = 5

    # lightly shade region that is not actually shown in model manifold
    alpha = 0.0
    y0s_colors = y0s_colormap.to_rgba(y0g)
    for i in range(npts):
        for j in range(npts):
            if y0g[i,j] > np.exp(t1/epsg[i,j]):
                y0s_colors[i,j] = gray_color


    # ax.imshow(np.tanh(y0g), extent=(1e-3,1e3,0,2), cmap='viridis', norm=y0s_colornorm) # show coloring of y0
    ax.imshow(y0s_colors, extent=(1e-3,1e3,0,2)) # show coloring of y0
    # add lines of constant eps
    for i, eps in enumerate(epss_samples):
        # only plot up to boundary
        count = y0s[y0s < np.exp(t1/eps)].shape[0]
        if i == 0 or i == epss_samples.shape[0]-1:
            ax.plot(eps*np.ones(count), y0s[-count:], lw=2.5*lw, color=eps_colors[i])
        else:
            ax.plot(eps*np.ones(count), y0s[-count:], lw=lw, color=eps_colors[i])

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['0', '1', str(y0max)])
    ax.set_xscale('log')
    ax.set_xticks([1e-3, 1e0, 1e3])
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$x_0$')
    ax.tick_params(axis='x', pad=10) # padding for xticks

    fig.subplots_adjust(bottom=0.15, left=0.07, right=0.93)

    plt.show()


def linear_sing_pert_twod_cmap_fig():
    """Plotting model manifold of y' = -1/eps * y"""

    cmap ='viridis'

    gray_color = np.array((0.05, 0.05, 0.05, 0.1))
    
    gsize = 40
    gspec = gs.GridSpec(gsize, gsize)

    t1, t2, t3 = (1.,2.,6.)# np.linspace(1,3,3)
    # constants for parameter space plot, also needed when making model manifold
    y0max = 2

    npts = 200
    xss = np.linspace(0,1,npts)
    yss = np.linspace(0,1,npts)
    xg, yg = np.meshgrid(xss,yss)
    xkept = np.empty(npts*npts)
    ykept = np.empty(npts*npts)
    count = 0
    origin = None
    for i in range(npts):
        for j in range(npts):
            if yg[i,j] <= xg[i,j] and xg[i,j] <= y0max*np.exp(-t1*np.log(y0max/yg[i,j])/t2):
                # keep track of which point is the origin for subsequent settings of y0, eps and z to 0 or inf
                if xg[i,j] == 0 and yg[i,j] == 0:
                    origin = count
                xkept[count] = xg[i,j]
                ykept[count] = yg[i,j]
                count = count + 1
    xkept = xkept[:count]
    ykept = ykept[:count]

    z = np.power(xkept, (t2 - t3)/(t2 - t1))*np.power(ykept, (t3 - t1)/(t2 - t1))
    z[origin] = 0 # z(0,0) = 0

    epss = (t2 - t1)/np.log(xkept/ykept)
    epss[np.isnan(epss)] = np.inf # eps(0,0,0) = inf
    # log_epssinv = np.log10(1./epss)
    epss_transformed = np.tanh(epss)

    y0s = xkept*np.power(xkept/ykept, t1/(t2 - t1))
    y0s[origin] = 0 # y0(origin) = 0
    # y0s[np.isnan(y0s)] = np.inf
    y0s_transformed = np.tanh(y0s)

    # construct two-dim colormap
    twod_cmap = TwoD_CMAP(np.array((0.,0.,1.,0.5)), np.min(epss_transformed), np.max(epss_transformed), np.array((1.,0.,0.,0.5)), np.min(y0s), np.max(y0s))
    # print np.min(log_epssinv), np.max(log_epssinv), np.array((1.,0.,0.,0.5)), np.min(y0s), np.max(y0s)

    xyz = np.array((xkept, ykept, z)).T

    tris = tri.Triangulation(xkept, ykept).triangles
    triangle_vertices = np.array([np.array((xyz[T[0]], xyz[T[1]], xyz[T[2]])) for T in tris]) # plot in k1, k-1, k2

    y0s_avg = calc_tri_avg(tris, y0s)
    epss_transformed_avg = calc_tri_avg(tris, epss_transformed)
    face_colors = twod_cmap.to_rgba(epss_transformed_avg, y0s_avg)

    coll_y0 = Poly3DCollection(triangle_vertices, linewidths=0)
    coll_y0.set_facecolor(face_colors)
    coll_y0.set_edgecolor(face_colors)
    # coll_eps = Poly3DCollection(triangle_vertices, facecolors=eps_colormap.to_rgba(calc_tri_avg(tris, epss_transformed)), linewidths=0, edgecolors='w')

    # add greyed-area that doesn't correspond to region in parameter space
    xkept = np.empty(npts*npts)
    ykept = np.empty(npts*npts)
    count = 0
    origin = None
    arb_tol = 0.00
    for i in range(npts):
        for j in range(npts):
            if yg[i,j] <= xg[i,j] and xg[i,j] >= y0max*np.exp(-t1*np.log(y0max/yg[i,j])/t2) - arb_tol:
                # keep track of which point is the origin for subsequent settings of y0, eps and z to 0 or inf
                if xg[i,j] == 0 and yg[i,j] == 0:
                    origin = count
                xkept[count] = xg[i,j]
                ykept[count] = yg[i,j]
                count = count + 1
    xkept = xkept[:count]
    ykept = ykept[:count]

    # plt.scatter(xkept, ykept)
    # plt.show()

    z = np.power(xkept, (t2 - t3)/(t2 - t1))*np.power(ykept, (t3 - t1)/(t2 - t1))
    z[origin] = 0 # z(0,0) = 0

    xyz = np.array((xkept, ykept, z)).T
    tris = tri.Triangulation(xkept, ykept).triangles
    triangle_vertices = np.array([np.array((xyz[T[0]], xyz[T[1]], xyz[T[2]])) for T in tris]) # plot in k1, k-1, k2

    face_colors_grey = np.ones((tris.shape[0],4))*gray_color
    face_y1s = calc_tri_avg(tris, xkept)
    face_y2s = calc_tri_avg(tris, ykept)
    arb_tol = 0.00
    for i in range(tris.shape[0]):
        if face_y1s[i] < y0max*np.exp(-t1*np.log(y0max/face_y2s[i])/t2) - arb_tol:
            face_colors_grey[i] = 0

    coll_y0_grey = Poly3DCollection(triangle_vertices, linewidths=0)
    coll_y0_grey.set_facecolor(face_colors_grey)
    # coll_y0_grey.set_edgecolor((1,1,1,1))
    # coll_y0_grey.set_edgecolor(gray_color)

    
    # plot both greyed and colored surface
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,gsize/2:], projection='3d')
    ax.add_collection(coll_y0)
    ax.add_collection(coll_y0_grey)

    # ax_cb = fig.add_subplot(gspec[:,gsize-1])
    # cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=y0s_colornorm, ticks=[0,1.,y0max], label=r'$x_0$')
    # cb.ax.set_yticklabels(['0', '1', str(y0max)])

    # add lines of constant epsilon
    epss_samples = np.array((np.min(epss), 1, 3.5, 10, 1e3)) # hand picked for nice visuals
    # eps_colornorm = colors.Normalize(vmin=np.min(np.log10(epss_samples)), vmax=1.5*np.max(np.log10(epss_samples))) # scale vmax by 1.5 to remove whiter part of cmap from use
    # eps_colormap = cm.ScalarMappable(norm=eps_colornorm, cmap='Oranges_r')
    eps_colors = ['#002255', '#0044aa', '#0066ff', '#5599ff', '#aaccff'] # ['#000000', '#0000cc', '#0099ff', '#cc00ff', '#ff0000']
    lw = 4
    for i, eps in enumerate(epss_samples):
        xss = np.linspace(0,np.min((1, y0max*np.exp(-t1/eps))),npts)
        yss = xss*np.exp((t1 - t2)/eps)
        zss = np.power(xss, (t2 - t3)/(t2 - t1))*np.power(yss, (t3 - t1)/(t2 - t1))
        ax.plot(xss, yss, zss, lw=lw, color=eps_colors[i])#eps_colormap.to_rgba(np.log10(eps)))

    ax.set_xlabel('\n' + r'$f_1$')
    ax.set_ylabel('\n' + r'$f_2$')
    ax.set_zlabel('\n' + r'$f_3$')
    formatter = FormatAxis(ax)
    ax.set_xticks((0,1))
    ax.set_yticks((0,1))
    ax.set_zticks((0,1))
    # formatter.format('x', xkept, '%1.0f', nticks=2)
    # formatter.format('y', ykept, '%1.0f', nticks=2)
    # formatter.format('z', z[::-1], '%1.0f', nticks=2)
    # ax.invert_zaxis()
    ax.invert_yaxis()


    # # plot y0, eps plane
    npts = 1000
    y0s = np.linspace(y0max,0,npts) # reverse ordering to get meshgrid output correctly oriented
    logeps_min = np.log10(np.min(epss))
    logeps_max = 3
    epss = np.logspace(logeps_min, logeps_max, npts)
    epsg, y0g = np.meshgrid(epss, y0s)

    ax = fig.add_subplot(gspec[:,:gsize/2-2])
    lw = 5

    # lightly shade region that is not actually shown in model manifold
    alpha = 0.0
    # y0s_colors = y0s_colormap.to_rgba(y0g)
    grid_colors = twod_cmap.to_rgba(np.tanh(epsg), y0g)
    for i in range(npts):
        for j in range(npts):
            if y0g[i,j] > np.exp(t1/epsg[i,j]):
                grid_colors[i,j] = gray_color


    ax.imshow(grid_colors, extent=(np.min(epss),np.max(epss),np.min(y0s),np.max(y0s))) # show coloring of y0
    # add lines of constant eps
    for i, eps in enumerate(epss_samples):
        # only plot up to boundary
        count = y0s[y0s < np.exp(t1/eps)].shape[0]
        if i == 0 or i == epss_samples.shape[0]-1:
            ax.plot(eps*np.ones(count), y0s[-count:], lw=2.5*lw, color=eps_colors[i])
        else:
            ax.plot(eps*np.ones(count), y0s[-count:], lw=lw, color=eps_colors[i])

    ax.set_yticks(np.linspace(0,y0max,3))
    # ax.set_yticklabels(['0', '1', str(y0max)])
    ax.set_xscale('log')
    ax.set_xticks(np.logspace(0,3,4))
    ax.set_xlim(left=np.min(epss))
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$x_0$')
    ax.tick_params(axis='x', pad=10) # padding for xticks

    fig.subplots_adjust(bottom=0.15, left=0.07)

    plt.show()    


class DMAPS_Data_Kernel:
    """Computes kernel between two points in parameter space, taking into account both the euclidean distance between parameters and the euclidean distance between model predictions at those parameters"""
    def __init__(self, epsilon, lam):
        self._epsilon = epsilon
        self._lam = lam

    def __call__(self, x1, x2):
        """Custom kernel given by: :math:`k(x_1, x_2) = e^{\frac{-(\| x_1 - x_2 \|^2 + \|m(x_1) - m(x_2)\|^2)}{\epsilon^2} - \frac{(of(x_1) - of(x_2))^2}{\epsilon}}` where :math:`m(x_i)` is the model prediction at parameter set :math:`x_i`

        Args:
            x1 (array): first data point in which x = [(parameters), (predictions)]
            x2 (array): second data point in which x = [(parameters), (predictions)]
        """
        return np.exp(-(np.power(np.linalg.norm(x1[0] - x2[0])/self._epsilon,2) + np.power(np.linalg.norm(x1[1] - x2[1]), 2))/(self._lam*self._lam))


def outlier_plot():
    """Shows the breakdown of k_eff in certain regions of parameter space, leading to "outliers" at each value of c(theta)"""
    data = np.load('../rawlings_model/data/params-ofevals.pkl')

    of_max = 1e-3
    data = data[data[:,0] < of_max]
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    print np.min(keff), np.max(keff)
    log_data = np.log10(data)
    print log_data.shape
    npts = 2500
    slicesize = log_data.shape[0]/npts
    data = data[::slicesize]
    log_data = log_data[::slicesize]
    keff = keff[::slicesize]
    npts = data.shape[0]

    # load data from data_dmaps_rawlings_fig()
    print 'loading saved trajectories'
    eigvals = np.load('./data/tempeigvals.pkl')
    eigvects = np.load('./data/tempeigvects.pkl')

    # # first plot c(theta) vs keff to show that the outliers exist at all values of c(theta), and that this phenomenon is a result of the model structure and not some choice of the DMAPS kernel
    gsize = 20
    gspec = gs.GridSpec(gsize, gsize)
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,:18])
    ax.scatter(keff, data[:,0], c=eigvects[:,1], s=100)
    ax.set_xlabel(r'$k_{eff}$')
    ax.set_ylabel(r'$\| \mu(\theta) - \mu(\theta^*) \|^2$')
    ax.set_xlim((0.46, 0.55))
    ax.set_ylim((0,0.001))
    axcb = fig.add_subplot(gspec[:,19])
    eigmin = np.min(eigvects[:,1])
    eigmax = np.max(eigvects[:,1])
    cb = colorbar.ColorbarBase(axcb, norm=colors.Normalize(eigmin, eigmax), ticks=np.linspace(eigmin, eigmax, 5), format='%1.3f', label=r'$\Phi_1$')

    # plain plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(keff, data[:,0], c='b', s=100)
    ax.set_xlabel(r'$k_{eff}$')
    ax.set_ylabel(r'$\| \mu(\theta) - \mu(\theta^*) \|^2$')
    ax.set_xlim((0.46, 0.55))
    ax.set_ylim((0,0.001))
    plt.show()


    # now throw some shit out
    tokeep = eigvects[:,1] < 0.01
    tothrow = tokeep == False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(keff[tokeep], data[tokeep,0], c='b', s=100)
    ax.scatter(keff[tothrow], data[tothrow,0], c='r', s=100)
    ax.set_xlabel(r'$k_{eff}$')
    ax.set_ylabel(r'$\| \mu(\theta) - \mu(\theta^*) \|^2$')
    ax.set_xlim((0.46, 0.55))
    ax.set_ylim((0,0.001))
    plt.show()
    

    # axcb = fig.add_subplot(gspec[:,39])
    # colornorm = colors.Normalize(vmin=keffmin, vmax=keffmax)
    # colorbar.ColorbarBase(axcb, norm=colornorm, ticks=np.linspace(keffmin, keffmax, 5), format='%1.2f', label=r'$k_{eff}$')

    
    # ax.set_ylim((0.46, 0.55))
    # ax.set_xlim((-0.035, 0.035))
    ax.set_xticks(np.linspace(-0.03, 0.03, 5))
    fig.subplots_adjust(bottom=0.15)

    # # now do the 3d stuff
    keffmin = np.min(keff)
    keffmax = np.max(keff)
    gsize = 40
    gspec = gs.GridSpec(gsize,gsize)

    # colored by keff
    # fig = plt.figure()
    # ax = fig.add_subplot(gspec[:,:37], projection='3d')
    # ax.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=keff)
    # axcb = fig.add_subplot(gspec[:,39])
    # colornorm = colors.Normalize(vmin=keffmin, vmax=keffmax)
    # colorbar.ColorbarBase(axcb, norm=colornorm, ticks=np.linspace(keffmin, keffmax, 5), format='%1.2f', label=r'$k_{eff}$')
    
    # formatter = FormatAxis(ax)
    # formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
    # formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
    # formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
    # ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    # ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    # ax.set_zlabel('\n\n' + r'$\log(k_2)$')

    # colored by keff
    fig = plt.figure()
    axk = fig.add_subplot(111, projection='3d')
    axk.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=keff, s=100)

    # colored by keff valid, keff invalid
    fig = plt.figure()
    ax0 = fig.add_subplot(111, projection='3d')
    ax0.scatter(log_data[tothrow,1], log_data[tothrow,2], log_data[tothrow,3], c='r')
    ax0.scatter(log_data[tokeep,1], log_data[tokeep,2], log_data[tokeep,3], c='b')

    # colored by phi1
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,1], s=100)

    # colored by phi2
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,2], s=100)

    # colored by phi3
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,3], s=100)

    for ax in [ax0, ax1, ax2, ax3, axk]:
        formatter = FormatAxis(ax)
        formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
        formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
        formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
        ax.set_xlabel('\n\n' + r'$\log(k_1)$')
        ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
        ax.set_zlabel('\n\n' + r'$\log(k_2)$')

    plt.show()    

def data_dmaps_rawlings_fig():
    """Plotting data dmaps output of rawlings model"""
    data = np.load('../rawlings_model/data/params-ofevals.pkl')

    of_max = 1e-3
    data = data[data[:,0] < of_max]
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    print np.min(keff), np.max(keff)
    log_data = np.log10(data)
    print log_data.shape
    npts = 2500
    slicesize = log_data.shape[0]/npts
    data = data[::slicesize]
    log_data = log_data[::slicesize]
    keff = keff[::slicesize]
    npts = data.shape[0]

    # calculate trajectories at all the kept points
    # settings taken from sample_param_grid_mpi in rawlings_model
    A0 = 1.0 # initial concentration of A
    k1_true = 0.1
    kinv_true = 1000.0
    k2_true = 1000.0
    decay_rate = k1_true*k2_true/(kinv_true + k2_true) # effective rate constant that governs exponential growth rate
    # start at t0 = 0, end at tf*decay_rate = 4
    eigval_plus_true = 0.5*(-(kinv_true + k1_true + k2_true) + np.sqrt(np.power(k1_true + kinv_true + k2_true, 2) - 4*k1_true*k2_true))
    dt = 0.5/np.abs(eigval_plus_true)
    # start at dt, end at 5*dt
    ntimes = 5 # somewhat arbitrary
    times = np.linspace(dt, 5*dt, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=False)
    trajectories = np.empty((npts, ntimes))
    print 'computing trajectories'
    for i, params in enumerate(data[:,1:]):
        trajectories[i] = model.gen_timecourse(*params)

    # # for setting lambda:
    # max_traj_dist = 0
    # for i in range(npts):
    #     for j in range(npts):
    #         if np.linalg.norm(trajectories[i] - trajectories[j]) > max_traj_dist:
    #             max_traj_dist = np.linalg.norm(trajectories[i] - trajectories[j])
    # print 'max distance between trajectories:', max_traj_dist
    # print 'saving trajectories'
    # trajectories.dump('./data/temptrajs.pkl')

    eigvals = None
    eigvects = None
    have_data = True
    if have_data:
        print 'loading saved trajectories'
        eigvals = np.load('./data/tempeigvals.pkl')
        eigvects = np.load('./data/tempeigvects.pkl')
    else:
        full_data = zip(log_data[:,1:], trajectories)

        # # working settings
        # epsilon = 100
        # lam = 1
        # also working settings (recovers all three relevant directions!)
        epsilon = 50
        lam = 0.1
        k = 12
        kernel = Data_Kernel(epsilon, lam)
        print 'starting data dmaps on', log_data.shape[0], 'points'
        eigvals, eigvects = dmaps.embed_data_customkernel(full_data, k, kernel)
        # eigvals, eigvects = dmaps.embed_data(log_data[:,1:], k, epsilon=epsilon)
        print 'completed data dmaps'
        eigvals.dump('./data/tempeigvals.pkl')
        eigvects.dump('./data/tempeigvects.pkl')

    # # first straight up plot phi1 vs keff
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(eigvects[:,1], keff, s=100)
    ax.set_xlabel(r'$\Phi_1$')
    ax.set_ylabel(r'$k_{eff}$')
    ax.set_ylim((0.045, 0.055))
    ax.set_xlim((-0.04, 0.04))
    # ax.set_xticks(np.linspace(-0.03, 0.03, 5))
    # fig.subplots_adjust(bottom=0.15)

    # # now do the 3d stuff
    keffmin = np.min(keff)
    keffmax = np.max(keff)
    gsize = 40
    gspec = gs.GridSpec(gsize,gsize)
    # colored by keff
    # fig = plt.figure()
    # ax = fig.add_subplot(gspec[:,:37], projection='3d')
    # ax.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=keff)
    # axcb = fig.add_subplot(gspec[:,39])
    # colornorm = colors.Normalize(vmin=keffmin, vmax=keffmax)
    # colorbar.ColorbarBase(axcb, norm=colornorm, ticks=np.linspace(keffmin, keffmax, 5), format='%1.2f', label=r'$k_{eff}$')
    
    # formatter = FormatAxis(ax)
    # formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
    # formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
    # formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
    # ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    # ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    # ax.set_zlabel('\n\n' + r'$\log(k_2)$')

    # colored by phi1
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,:37], projection='3d')
    ax.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,1], s=100)
    axcb = fig.add_subplot(gspec[:,39])
    colornorm = colors.Normalize(vmin=np.min(eigvects[:,1]), vmax=np.max(eigvects[:,1]))
    colorbar.ColorbarBase(axcb, norm=colornorm, ticks=np.linspace(-0.03, 0.03, 5), format='%1.3f', label=r'$\Phi_1$')


    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # now plot transparencies of keff = constant
    # colornorm = colors.Normalize(vmin=np.log10(keffmin), vmax=np.log10(keffmax))
    # colormap = cm.ScalarMappable(norm=colornorm, cmap='autumn')
    # k1min, k1max = np.min(data[:,1]), np.max(data[:,1])
    # kinvmin, kinvmax = np.min(data[:,2]), np.max(data[:,2])
    # k2min, k2max = np.min(data[:,3]), np.max(data[:,3])
    # true_data = data[np.argsort(data[:,0])[0]] # pull out parameter combo that has minimum obj. fn. value
    # keff_true = true_data[1]*true_data[3]/(true_data[2] + true_data[3])
    # npts = 200
    # stride = 2
    # kinvs, k2s = np.meshgrid(np.logspace(np.log10(kinvmin), np.log10(kinvmax), npts), np.logspace(np.log10(k2min), np.log10(k2max), npts))
    # nkeffs = 2
    # alpha = 0.3
    # orange = np.array((255., 170., 18., alpha*255.))/255
    # red = np.array((255., 97., 18., alpha*255.))/255
    # orange_red = np.array((red, orange))
    # for k, keff in enumerate(np.logspace(np.log10(keffmin) + 0.1*(np.log10(keffmax) - np.log10(keffmin)), np.log10(keffmax) - 0.2*(np.log10(keffmax) - np.log10(keffmin)), nkeffs)):
    #     k1s = keff*(kinvs + k2s)/k2s
    #     # kinvs = k1s*k2s/keff - k2s
    #     cs = np.ones((npts, npts, 4))*orange_red[k] # np.ones((npts, npts, 4))*colormap.to_rgba(np.log10(keff))*np.array((1,1,1,alpha)) # change color and alpha
    #     for i in range(npts):
    #         for j in range(npts):
    #             # if kinvs[i,j] < 10: # np.log10(kinvs[i,j] < 1): # but have kinvs < 0
    #             #     cs[i,j,3] = 0
    #             if k1s[i,j] > .99:
    #                 cs[i,j,3] = 0

    #     # ax.plot_surface(np.log10(k1s), np.log10(kinvs), np.log10(k2s), linewidth=0, facecolors=cs, edgecolors=np.array((0,0,0,0)), shade=False, cstride=stride, rstride=stride, antialiased=True)
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    formatter = FormatAxis(ax)
    formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
    formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
    formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')


    plt.show()


def dmaps_rawlings_fig():
    """Plotting dmaps output of rawlings model - AZ visit"""
    data = np.load('../rawlings_model/data/params-ofevals.pkl')

    of_max = 1e-6
    data = data[data[:,0] < of_max]
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    log_data = np.log10(data)
    npts = 2500
    slicesize = log_data.shape[0]/npts
    if slicesize <  1:
        slicesize = 1
    data = data[::slicesize]
    log_data = log_data[::slicesize]
    keff = keff[::slicesize]
    npts = data.shape[0]
    print 'have', npts, 'pts in dataset'


    epsilon = 3.0
    k = 12 # number of dimensions for embedding

    have_data = True
    if have_data:
        print 'loading old data from ./temp.pkl'
        eigvects = np.load('./temp.pkl')
    else:
        print 'starting dmaps'
        eigvals, eigvects = dmaps.embed_data(log_data[:,1:], k, epsilon=epsilon)
        eigvects.dump('./temp.pkl')
        print 'completed dmaps'


    alpha = 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,1], s=100, alpha=alpha)


    formatter = FormatAxis(ax)
    formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
    formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
    formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(log_data[:,1], log_data[:,2], log_data[:,3], c=eigvects[:,2], s=100, alpha=alpha)



    formatter = FormatAxis(ax)
    formatter.format('x', log_data[:,1], '%1.1f', nticks=3)
    formatter.format('y', log_data[:,2], '%1.1f', nticks=3)
    formatter.format('z', log_data[:,3], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')

    plt.show()


def rawlings_keff_fig():
    """Plots level sets/sloppy manifold of rawlings model when k1 << k2, k1 << kinv, showing that keff changes in a direction normal to the surface"""

    data = np.load('../rawlings_model/data/params-ofevals.pkl')

    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot fat surface with of=1e-3, with three surfaces mapping out different level sets
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    of_max = 1e-3
    data = data[data[:,0] < of_max]
    log_data = np.log10(data)
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    print 'shape of data after trimmming:', data.shape

    # set up colorbar based on log(keff)
    keffmin, keffmax = np.min(keff), np.max(keff)
    print 'keff range:', keffmin, keffmax
    colornorm = colors.Normalize(vmin=np.min(log_data[:,0]), vmax=np.max(log_data[:,0])) # colors.Normalize(vmin=np.log10(keffmin), vmax=np.log10(keffmax))
    cmap = 'viridis'
    colormap = cm.ScalarMappable(norm=colornorm, cmap=cmap)

    colornorm_keff = colors.Normalize(vmin=np.log10(keffmin), vmax=np.log10(keffmax))
    colormap_keff = cm.ScalarMappable(norm=colornorm_keff, cmap=cmap)


    gsize = 10
    gspec = gs.GridSpec(gsize,gsize)
    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    slice = data.shape[0]/20000
    # scatter = ax.scatter(log_data[::slice,1], log_data[::slice,2], log_data[::slice,3], c=colormap.to_rgba(np.log10(keff[::slice])), alpha=0.5)
    scatter = ax.scatter(log_data[::slice,1], log_data[::slice,2], log_data[::slice,3], c=colormap.to_rgba(log_data[::slice,0]), alpha=0.5)
    

    # now plot transparencies of keff = constant
    k1min, k1max = np.min(data[:,1]), np.max(data[:,1])
    kinvmin, kinvmax = np.min(data[:,2]), np.max(data[:,2])
    k2min, k2max = np.min(data[:,3]), np.max(data[:,3])
    true_data = data[np.argsort(data[:,0])[0]] # pull out parameter combo that has minimum obj. fn. value
    keff_true = true_data[1]*true_data[3]/(true_data[2] + true_data[3])
    npts = 200
    stride = 2
    kinvs, k2s = np.meshgrid(np.logspace(np.log10(kinvmin), np.log10(kinvmax), npts), np.logspace(np.log10(k2min), np.log10(k2max), npts))
    nkeffs = 2
    alpha = 0.5
    for keff in np.logspace(np.log10(keffmin), np.log10(keffmax), nkeffs):
        k1s = keff*(kinvs + k2s)/k2s
        # kinvs = k1s*k2s/keff - k2s
        cs = np.ones((npts,npts,4))*np.array((204./255.,0,0,alpha)) # np.ones((npts, npts, 4))*colormap_keff.to_rgba(np.log10(keff))*np.array((1,1,1,alpha)) # change color and alpha
        for i in range(npts):
            for j in range(npts):
                # if kinvs[i,j] < 10: # np.log10(kinvs[i,j] < 1): # but have kinvs < 0
                #     cs[i,j,3] = 0
                if k1s[i,j] > .099:
                    cs[i,j,3] = 0

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(np.log10(k1s), np.log10(kinvs), np.log10(k2s), linewidth=0, facecolors=cs, edgecolors=np.array((0,0,0,0)), shade=False, cstride=stride, rstride=stride, antialiased=True)

    ax.set_ylim((np.log10(kinvmin), np.log10(kinvmax)))
    formatter = FormatAxis(ax)
    formatter.format('x', log_data[::slice,1], '%1.1f', nticks=3)
    formatter.format('y', log_data[::slice,2], '%1.1f', nticks=3)
    formatter.format('z', log_data[::slice,3], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')
    ax_cb = fig.add_subplot(gspec[:,gsize-1])
    ax_cb.text(-0.1, 1.05, r'$\log(\delta(\theta))$', transform=ax_cb.transAxes, fontsize=48) # r'$k_{eff}$', transform=ax_cb.transAxes, fontsize=48)
    # colornorm = colors.Normalize(vmin=keffmin, vmax=keffmax)
    print np.linspace(np.min(log_data[:,0]), np.max(log_data[:,0]), 5)
    colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colornorm, ticks=np.linspace(np.min(log_data[:,0]), np.max(log_data[:,0]), 5), format='%1d')#, label=r'$k_{eff}$')
    plt.show()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # now do the same with of = 1e-6, only include one surface
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    of_max = 1e-6
    data = data[data[:,0] < of_max]
    log_data = np.log10(data)
    keff = data[:,1]*data[:,3]/(data[:,2] + data[:,3])
    print 'shape of data after trimmming:', data.shape

    # set up colorbar based on log(keff)
    keffmin, keffmax = np.min(keff), np.max(keff)
    print 'keff range:', keffmin, keffmax
    colornorm = colors.Normalize(vmin=np.min(log_data[:,0]), vmax=np.max(log_data[:,0]))
    colormap = cm.ScalarMappable(norm=colornorm, cmap=cmap)

    colornorm_keff = colors.Normalize(vmin=np.log10(keffmin), vmax=np.log10(keffmax))
    colormap_keff = cm.ScalarMappable(norm=colornorm, cmap=cmap)

    fig = plt.figure()
    ax = fig.add_subplot(gspec[:,:gsize-1], projection='3d')
    slice = 1
    # scatter = ax.scatter(log_data[::slice,1], log_data[::slice,2], log_data[::slice,3], c=colormap.to_rgba(np.log10(keff[::slice])), alpha=0.2)
    scatter = ax.scatter(log_data[::slice,1], log_data[::slice,2], log_data[::slice,3], c=colormap.to_rgba(log_data[::slice,0]), alpha=0.2)
    

    # now plot transparencies of keff = keff_true
    k1min, k1max = np.min(data[:,1]), np.max(data[:,1])
    kinvmin, kinvmax = np.min(data[:,2]), np.max(data[:,2])
    k2min, k2max = np.min(data[:,3]), np.max(data[:,3])
    true_data = data[np.argsort(data[:,0])[0]] # pull out parameter combo that has minimum obj. fn. value
    keff_true = true_data[1]*true_data[3]/(true_data[2] + true_data[3])
    npts = 100
    stride = 1
    kinvs, k2s = np.meshgrid(np.logspace(np.log10(kinvmin), np.log10(kinvmax), npts), np.logspace(np.log10(k2min), np.log10(k2max), npts))
    alpha = 0.2
    # hard coded keff for nice color
    # kinvs = k1s*k2s/keff - k2s
    keff = 0.502
    k1s = keff_true*(kinvs + k2s)/k2s
    cs = np.ones((npts,npts,4))*np.array((204./255.,0,0,alpha))
    # cs = np.ones((npts, npts, 4))*colormap.to_rgba(np.log10(keff))*np.array((1,1,1,alpha)) # change color and alpha
    for i in range(npts):
        for j in range(npts):
            if k1s[i,j] > .099:
                cs[i,j,3] = 0


    ax.plot_surface(np.log10(k1s), np.log10(kinvs), np.log10(k2s), linewidth=0, facecolors=cs, edgecolors=np.array((0,0,0,0)), shade=False, cstride=stride, rstride=stride, antialiased=True)

    ax.set_ylim((np.log10(kinvmin), np.log10(kinvmax)))
    formatter = FormatAxis(ax)
    formatter.format('x', log_data[::slice,1], '%1.1f', nticks=3)
    formatter.format('y', log_data[::slice,2], '%1.1f', nticks=3)
    formatter.format('z', log_data[::slice,3], '%1.1f', nticks=3)
    ax.set_xlabel('\n\n' + r'$\log(k_1)$')
    ax.set_ylabel('\n\n' + r'$\log(k_{-1})$')
    ax.set_zlabel('\n\n' + r'$\log(k_2)$')
    ax_cb = fig.add_subplot(gspec[:,gsize-1])
    # colornorm = colors.Normalize(vmin=keffmin, vmax=keffmax)
    ax_cb.text(-0.1, 1.05, r'$\log(\delta(\theta))$', transform=ax_cb.transAxes, fontsize=48)
    # ax_cb.text(0.2, 1.05, r'$k_{eff}$', transform=ax_cb.transAxes, fontsize=48)
    colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colornorm, ticks=np.linspace(np.min(log_data[:,0]), np.max(log_data[:,0]), 5), format='%1d')#, label=r'$k_{eff}$') # colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=colornorm, ticks=np.linspace(keffmin, keffmax, 5), format='%1.3f')
    plt.show()


def gear_data_dmaps():
    """trying to show that at the appropriate scale, model manifold appears one-dimensional, or at least the skinny second dimension is pushed further down in the DMAPS spectrum"""
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # constants are from rawlings_model/main.py sampling routine, take care to match them manually
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    A0 = 1.0 # initial concentration of A
    k1_true = 0.1
    kinv_true = 1000.0
    k2_true = 1000.0
    eigval_plus_true = 0.5*(-(kinv_true + k1_true + k2_true) + np.sqrt(np.power(k1_true + kinv_true + k2_true, 2) - 4*k1_true*k2_true))
    dt = 0.5/np.abs(eigval_plus_true)
    # start at dt, end at 5*dt
    ntimes = 5 # somewhat arbitrary
    times = np.linspace(dt, 5*dt, ntimes)
    model = Rawlings_Model(times, A0, k1_true, kinv_true, k2_true, using_sympy=False)

    data = np.load('../rawlings_model/data/params-ofevals.pkl')
    ofs = [1e-5, 1e-3, 1e-1]
    npts = 4000
    labels = ['a', 'b', 'c']
    for i, of in enumerate(ofs):

        # calculate dmap
        trimmed_data = data[data[:,0] < of]
        slice = trimmed_data.shape[0]/npts
        # slice data and remove obj. fn. evals in first column
        trimmed_data = trimmed_data[::slice,1:]
        npts = trimmed_data.shape[0]
        trajs = np.empty((npts, ntimes))
        for j, k in enumerate(trimmed_data):
            trajs[j] = model.gen_timecourse(k[0], k[1], k[2])
        print 'calculating DMAP for index', i
        ndims = 50
        eigvals, eigvects = dmaps.embed_data(trajs, k=ndims, epsilon = np.sqrt(of)/8)
        eigvects.dump('./data/temp' + str(i) + '.pkl')
        # eigvects = np.load('./data/temp' + str(i) + '.pkl')

        keffs = trimmed_data[:,0]*trimmed_data[:,2]/(trimmed_data[:,1] + trimmed_data[:,2])
        print keffs.shape, trimmed_data.shape, eigvects.shape
        keffs.dump('./data/tempkeff' + str(i) + '.pkl')
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hold(False)
        # for j in range(2,ndims-1):
        #     ax.scatter(eigvects[:,1], eigvects[:,j], c=keffs)
        #     plt.savefig('./figs/temp/dmap' + labels[i] + str(j) + '.png')


def main():
    # dmaps_2d_epsilon()
    # sing_pert_contours()
    # dmaps_line()
    # dmaps_2d_dataspace()
    # dmaps_1d_dataspace()
    # analytical_anisotropic_diffusion()
    # analytical_anisotropic_diffusion_eigenfns()
    # dmaps_plane()
    # sing_pert_data_space_fig_test2_gendata()
    # sing_pert_contours_pydstool()
    # sing_pert_data_space_fig_test()
    # rawlings_surface_normal()
    # rawlings_2d_dmaps_fig()
    # dmaps_rawlings_fig()
    # data_dmaps_rawlings_fig()
    # rawlings_keff_fig()
    # rawlings_3d_dmaps_fig()
    # two_effective_one_neutral_dmaps_fig()
    # discretized_laplacian_dmaps()
    # transformed_param_space_fig()
    rawlings_keff_delta_figs()
    # sing_pert_contours_fig1() # needs work perhaps, particularly to close the contour at 10^{-3}
    # sing_pert_contours_fig2()
    # test()
    # temp_fig()
    # linear_sing_pert_model_manifold_fig()
    # linear_sing_pert_twod_cmap_fig()
    # linear_sing_pert_param_space_log_fig()
    # linear_sing_pert_param_space_fig()
    # outlier_plot()
    # gear_data_dmaps()
    # rgb_square_test()
    
if __name__=='__main__':
    main()
