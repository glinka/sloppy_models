"""

Package requirements: NumPy, SciPy, Matplotlib

"""
# numpy (numerics)
import numpy as np
# scipy (more numerics)
import scipy.sparse.linalg as spla
# matplotlib (plotting)
from matplotlib import colors, colorbar, cm, pyplot as plt, gridspec as gs, tri


def main():
    """Computes a DMAP of dataset evenly spaced along a one-dimensional, piecewise-linear line. Useful for investigating effects of non-monotonicity on DMAPS output."""

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Adjustable settings:
        # a_true: slope of 1st and 3rd segment
        # b_true: slope of 2nd segments
        # lam_true: lambda used in DMAPS calculations (scaling of y-axis)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    a_true = 1.  # slope of 1st and 3rd segment
    b_true = -1  # slope of 2nd segment
    lam_true = 0.01 # the actual effective lambda used in calculations
    # note: we scale the dataset by 'lam_true' and regenerate evenly spaced samples along this new curve, thus avoiding any density-driven effects in the DMAPS output

    a = a_true/np.sqrt(lam_true) # the effective 'a' value based on a_true and lam_true
    b = b_true/np.sqrt(lam_true) #     ''        'b'            ''
    ra = np.sqrt(1+a*a) # arc-length factors on segments 
    rb = np.sqrt(1+b*b)  
    L = 2*ra/3 + rb/3  # total graph length
    N = 200  # no. of grid intervals of length ds
    ds = L/N  # arc length of grid interval

    xy = np.zeros((N+1,2))  # initialization of (x,y)-values
    i = 0  # counter initialization

    # # assign x and y values based on slopes
    # 1st segment
    while xy[i,0] < 1./3:
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)] 
        i = i+1 

    # 1st point of 2nd segment
    xy[i,0] = 1./3 + (ds - (1./3-xy[i-1,0])*ra)/rb
    xy[i,1] = xy[i-1,1] + a*(1./3 - xy[i-1,0]) + b*(xy[i,0] - 1./3) 

    # 2nd segment
    while xy[i,0] < 2./3:
        xy[i+1] = xy[i] + [ds/rb,ds*np.sqrt(1-1/(rb*rb))*np.sign(b)] 
        i = i+1 

    # 1st point of 3rd segment
    xy[i,0] = 2./3 + (ds - (2./3-xy[i-1,0])*rb)/ra 
    xy[i,1] = xy[i-1,1] + b*(2./3 - xy[i-1,0]) + a*(xy[i,0] - 2./3) 

    # 3rd segment (TOL needed to avoid floating point errors)
    TOL = 1e-12
    while xy[i,0] < (1-TOL):
        xy[i+1] = xy[i] + [ds/ra,ds*np.sqrt(1-1/(ra*ra))*np.sign(a)]
        i = i+1

    # actual arc lengths (measured in ds-units) two entries should be slightly
    # under one, due because we measure along chord instead of through corner.

    ds_ratio = np.empty(N)
    for i in range(N):
        ds_ratio[i] = np.sqrt(np.power(xy[i+1,0]-xy[i,0], 2) + np.power(xy[i+1,1]-xy[i,1], 2))/ds 

    # # print ds_ratio

    # # plot dataset:
    plot_dataset = True
    if plot_dataset:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xy[:,0], xy[:,1], lw=5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\gamma(x)$')
        ax.set_xlim((0,1))
        plt.show()


    # # calculate a stretched/compressed cos(pi x) curve that should match the DMAPS output
    scaled_cos = np.zeros(N+1)
    i = 0
    while xy[i,0] < 1./3:
        scaled_cos[i] = np.cos(np.pi*ra*xy[i,0]/L)
        i = i + 1
    while xy[i,0] < 2./3:
        scaled_cos[i] = np.cos(np.pi*rb*(xy[i,0] - (1 - ra/rb)/3)/L)
        i = i + 1
    for x in xy[i:]:
        scaled_cos[i] = np.cos(np.pi*ra*(xy[i,0] - (1 - rb/ra)/3)/L)
        i = i + 1

    # # do some dmaps with the new kernel
    npts = N + 1
    xvals = xy[:,0]
    yvals = xy[:,1]

    # matrix which will contain kernel evaluations
    M = np.empty((npts,npts))
    lam = 1.0 # as we've scaled the data, we use a fake lambda of '1' here, 'lam_true' stores the true, effective value used in the output
    eps = 2*ds
    for i in range(npts):
        x1 = xvals[i]
        y1 = yvals[i]
        for j in range(npts):
            x2 = xvals[j]
            y2 = yvals[j]
            M[i,j] = np.exp(-(np.power(x1-x2, 2) + np.power(y1 - y2, 2)/lam)/eps) # k(x_i, x_j)
    # normalize by degree
    D_half_inv = np.identity(npts)/np.sqrt(np.sum(M,1))

    # # find eigendecomp
    k = 50 # number of eigenpairs to calculate
    Meigvals, Meigvects = spla.eigsh(np.dot(np.dot(D_half_inv, M), D_half_inv), k=k)
    Meigvects = np.dot(D_half_inv, Meigvects)
    # # sort the output by decreasing eigenvalue
    sorted_indices = np.argsort(np.abs(Meigvals))[::-1]
    Meigvals = Meigvals[sorted_indices]
    Meigvects = Meigvects[:,sorted_indices]
    Meigvects = Meigvects/np.linalg.norm(Meigvects, axis=0)

    # # plot output
    gsize = 2
    gspec = gs.GridSpec(gsize,gsize)
    fig = plt.figure(figsize=(36, 20))
    ax = fig.add_subplot(gspec[0,:])
    # plot dataset colored by DMAPS phi1
    ax.scatter(xy[:,0], xy[:,1], c=np.sign(Meigvects[0,1])*Meigvects[:,1], s=100)
    ax.set_ylabel(r'$\gamma(x)$')

    ax2 = fig.add_subplot(gspec[1,:], sharex=ax)
    # plot both the scaled cos and the first dmaps eigenvector
    ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[0,1]*scaled_cos, c='r', label='Scaled cos(x)')
    ax2.plot(xy[:,0], np.sign(Meigvects[0,1])*Meigvects[:,1], c='b', label='DMAPS')

    # minor formatting
    ax2.legend(fontsize=36)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\Phi_1$')
    ax.set_xlim((0, np.max(xy[:,0])))

    # set title
    lamstr = '%1.1f' % lam_true
    bstr = '%1.2f' % b_true
    astr = '%1.2f' % a_true
    title = r' $b=$' + bstr
    ax.set_title(title)

    fig.subplots_adjust(bottom=0.15)

    plt.show()


if __name__=='__main__':
    main()
