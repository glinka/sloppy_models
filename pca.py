import numpy as np
import matplotlib.pyplot as plt

def pca(data, ncomp=None):
    # expect individual data vectors to
    # be input as rows in data, so
    # data.shape = (npoints, nvars)
    # ncomp represents the number of
    # components to store
    X = np.copy(np.array(data))
    npoints = X.shape[0]
    nvars = X.shape[1]

    # center the data at the origin
    # (trust the magic of broadcasting)
    X = X - np.average(X, 0)

    # an unset ncomp will default to 
    # returning the full transformation
    if ncomp == None:
        ncomp = nvars

    u, s, v = np.linalg.svd(X)
    # note the "-s" to reverse sorting order
    sorted_indices = np.argsort(-s)
    princ_vars = s[sorted_indices[:ncomp]]
    princ_comps = v[sorted_indices[:ncomp], :]
    # need to change shape to take transpose
    return princ_vars, np.transpose(princ_comps)
        
def test_pca():
    from mpl_toolkits.mplot3d import Axes3D

    # define a planar equation
    z = lambda x, y: 3*x - y + 4
    npoints = 100
    stdev = 0.5
    noise = stdev*np.random.normal(size=npoints)
    xvals = np.random.uniform(low=-1, high=1, size=npoints)
    yvals = np.random.uniform(low=-4, high=4, size=npoints)
    zvals = z(xvals, yvals) + noise

    data = np.transpose(np.array([xvals, yvals, zvals]))
    pvar, pcomp = pca(data)

    # plot projections along various components
    ncomp = pvar.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cs = ['b', 'r', 'g']
    data_avg = np.average(data, 0)
    centered_data = data - data_avg
    # do not bother plotting "full" projection, which would give back the original data
    for i in range(2):
        # need to add back the averages
        proj = np.dot(np.dot(centered_data, pcomp[:,:i+1]), np.transpose(pcomp[:,:i+1])) + data_avg
        ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=cs[i], lw=0)
        print str(i+1) + 'd projection norm: ', np.linalg.norm(data - proj)
    # plot original data
    ax.scatter(data[:,0], data[:,1], data[:,2], c='g', lw=0)
    plt.show(fig)
    
    

if __name__ == '__main__':
    test_pca()
