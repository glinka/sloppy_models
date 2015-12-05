import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from algorithms import Integration

def sing_pert_contours():
    """Plots contours of system used in 'sing_pert_data_space' in y0 and epsilon"""
    x0 = 1
    lam = 1
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.linspace(0.01, 0.31, 3) # for nonlinear eqns

    # generate true/base trajectory
    y0_true = 2
    eps_true = 1e-3
    yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/eps_true
    y_integrator = Integration.Integrator(yprime)
    traj_true = y_integrator.integrate(np.array((y0_true,)), times)[:,0]


    npts = 200
    y0s = np.linspace(-1, 3, npts)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # may help visually to set epss = np.logspace(-1, -6, npts), though the resulting plot is not practically useful for the paper
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # epss = np.logspace(3, -3, npts)
    epss = np.logspace(-3.5, -1, npts)
    y0mesh, epsmesh = np.meshgrid(y0s, epss)
    of_evals = np.empty((npts,npts))
    # record data space output at each value of (y0, eps)
    for i in range(npts):
        for j in range(npts):
            yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/epsmesh[i,j]
            y_integrator = Integration.Integrator(yprime)
            traj = y_integrator.integrate(np.array((y0mesh[i,j],)), times)[:,0]
            of_evals[i,j] = np.power(np.linalg.norm(traj - traj_true), 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.scatter(epsmesh, y0mesh, c=np.log10(of_evals), s=40)
    ax.set_xlim((np.min(epsmesh), np.max(epsmesh)))
    ax.set_ylim((np.min(y0mesh), np.max(y0mesh)))
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$y_0$')
    plt.show()

def sing_pert_data_space():
    """Plots three dimensional data space of the classic singularly perturbed ODE system: x' = -lambda*x, y' = -y/epsilon. Probably will end up plotting {y(t1), y(t2), y(t3)} though may also plot norms {|| (x(t1), y(t1)) ||, ... } though this doesn't represent data space in the traditional sense. In particular we're interested in observing the transition from 2 to 1 to 0 dimensional parameter -> data space mappings, i.e. 2 stiff to 1 stiff to 0 stiff parameters."""
    # define constants
    x0 = 1
    lam = 1
    # define times at which to sample data. Always take three points as this leads to three-dimensional data space
    # t0 = 0.01; tf = 3
    # times = np.linspace(t0,tf,3) # for linear eqns
    # x trajectory is constant as x0 and lambda are held constant
    times = np.linspace(0.01, 0.31, 3) # for nonlinear eqns
    xs = x0*np.exp(-times*lam)
    ys = lambda y0, eps: y0*np.exp(-times/eps)

    # # examine trajectories based on linear eqn. to visualize data
    # plt.plot(np.linspace(0, tf, 1000), x0*np.exp(-np.linspace(t0, tf, 1000)*lam), color='r') # x traj
    # plt.plot(np.linspace(0, tf, 1000), 3*np.exp(-np.linspace(t0, tf, 1000)/1e1), c='b') # y traj
    # plt.scatter(times, xs, s=15, c='r', label='x')
    # plt.scatter(times, ys(3, 1e1), s=15, c='b', label='y')
    # plt.legend()
    # plt.show()
    # # examine trajectories based on nonlinear ode
    # eps = 0.1
    # yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/eps
    # y_integrator = Integration.Integrator(yprime)
    # times = np.linspace(0.001, 0.5, 100)
    # for y0 in np.linspace(-1,5,4):
    #     traj = y_integrator.integrate(np.array((y0,)), times)
    #     plt.plot(times, traj)
    # plt.show()
    # # examine the zeros of the nonlinear rhs
    # plt.plot(np.linspace(-10,10,100), yprime(0, np.linspace(-10,10,100)))
    # plt.show()

    # set up grid of points to sample in y0/eps space
    npts = 50
    y0s = np.linspace(1, 3, npts)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # may help visually to set epss = np.logspace(-1, -6, npts), though the resulting plot is not practically useful for the paper
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    epss = np.logspace(0, -4, npts)
    data = np.empty((npts*npts, 5))
    count = 0
    # record data space output at each value of (y0, eps)
    for y0 in y0s:
        for eps in epss:

            yprime = lambda t, y: -y*(1+1/(2*np.sin(y)))/eps
            y_integrator = Integration.Integrator(yprime)
            data[count,:3] = y_integrator.integrate(np.array((y0,)), times)[:,0]

            # data[count,:3] = ys(y0, eps) # np.linalg.norm(np.array((xs, ys(y0, eps))), axis=0)
            data[count,3:] = (y0, eps)
            count = count + 1

    # plot data space results, colored by both y0 (first) and eps (second)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], lw=0)
    ax.set_xlim((0.99*np.min(data[:,0]), 1.01*np.max(data[:,0])))
    ax.set_ylim((0.99*np.min(data[:,1]), 1.01*np.max(data[:,1])))
    ax.set_zlim((0.99*np.min(data[:,2]), 1.01*np.max(data[:,2])))
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.set_zlabel(r'$y_3$')
    # ax.set_title('colored by y0')
    ax.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=0)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,4], lw=0)
    ax.set_xlim((0.99*np.min(data[:,0]), 1.01*np.max(data[:,0])))
    ax.set_ylim((0.99*np.min(data[:,1]), 1.01*np.max(data[:,1])))
    ax.set_zlim((0.99*np.min(data[:,2]), 1.01*np.max(data[:,2])))
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.set_zlabel(r'$y_3$')
    # ax.set_title('colored by eps')
    ax.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=0)
    plt.show()
            
def main():
    # sing_pert_data_space()
    sing_pert_contours()

if __name__=='__main__':
    main()
