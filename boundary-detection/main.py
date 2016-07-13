import sys, os
import subprocess
import tempfile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from pca import pca

hull_path = "./hull.exe"

def get_alpha_shape(points):
    # Write points to tempfile
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    np.savetxt(tmpfile, points, fmt='%0.12f')
    # for point in points:
    #     tmpfile.write("%0.7f %0.7f\n" % point)
    tmpfile.close()

    # Run hull
    command = "%s -A -m1000000 -oN < %s" % (hull_path, tmpfile.name)
    # command = "%s -A -oN < %s" % (hull_path, tmpfile.name)
    print >> sys.stderr, "Running command: %s" % command
    retcode = subprocess.call(command, shell=True)
    if retcode != 0:
        print >> sys.stderr, "Warning: bad retcode returned by hull.  Retcode value:" % retcode
    os.remove(tmpfile.name)

    # Parse results
    results_file = open("hout-alf")
    results_file.next() # skip header
    results_indices = np.array([[int(i) for i in line.rstrip().split()] for line in results_file])
    results_indices = results_indices.flatten()
    unique_indices = []
    for index in results_indices:
        if index not in unique_indices:
            unique_indices.append(index)
#    print "results length = %d" % len(results_indices)
    results_file.close()
    os.remove(results_file.name)

    return np.array(unique_indices) #np.array([points[i] for i in unique_indices])

def rawlings_test():
    # # test dis code with parameters from rawlings' model

    # xys = np.random.uniform(size=(500,2))
    # boundary = get_alpha_shape(xys)
    # print 'completed alpha-shape calculation'
    # print boundary.shape

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xys[:,0], xys[:,1])
    # ax.scatter(boundary[:,0], boundary[:,1], c='r')
    # plt.show()

    # # does work, try real data

    # data = np.load('../rawlings_model/data/params-ofevals.pkl')
    # of_max = 1e-3
    # data = data[data[:,0] < of_max]
    # slice = 25
    # data = data[::slice]
    # log_data = np.log10(data[:,1:])
    # print 'calculating alpha-shape with', log_data.shape, 'pts'
    # boundary = get_alpha_shape(log_data)
    # print 'finished alpha-shape calculation,', 1.0*boundary.shape[0]/log_data.shape[0], '% of pts included in boundary'

    data = np.load('../rawlings_model/data/params-ofevals.pkl')
    of_max = 1e-6
    data = data[data[:,0] < of_max]
    log_data = np.log10(data[:,1:])
    npts = log_data.shape[0]
    log_data_avg = np.average(log_data, axis=0)
    u, s, vT = np.linalg.svd(log_data - log_data_avg, compute_uv=True)
    twod_basis = vT[:2,:].T # first two rows of vT
    twod_embedding = np.dot(log_data, twod_basis)


    print 'calculating alpha-shape with', twod_embedding.shape, 'pts'
    boundary_indices = get_alpha_shape(twod_embedding)
    nboundary_pts = boundary_indices.shape[0]

    # calculate pairwise distances
    boundary_dist_matrix = np.zeros((nboundary_pts, nboundary_pts))
    for i in range(nboundary_pts):
        pt1 = log_data[boundary_indices[i]]
        for j in range(i+1,nboundary_pts):
            pt2 = log_data[boundary_indices[j]]
            boundary_dist_matrix[i,j] = np.linalg.norm(pt1 - pt2)
            boundary_dist_matrix[j,i] = boundary_dist_matrix[i,j]

    dist_matrix = np.zeros((nboundary_pts, npts))
    for i in range(nboundary_pts):
        pt1 = log_data[boundary_indices[i]]
        for j in range(npts):
            pt2 = log_data[j]
            dist_matrix[i,j] = np.linalg.norm(pt1 - pt2)

    # find neighbors in epsilon-ball of each boundary point and project outwards
    ball_radius = 0.3
    step_size = 0.1
    used_indices = []
    boundary_pts = np.copy(twod_embedding[boundary_indices])
    expanded_boundary_pts = np.empty((nboundary_pts, 3))
    for i in range(nboundary_pts):

        # find all points within 'ball_radius' of the current point on the boundary
        sorted_dist_indices = np.argsort(dist_matrix[i])
        j = 0
        new_boundary_indices = []
        while dist_matrix[i, sorted_dist_indices[j]] < ball_radius:
            if sorted_dist_indices[j] in boundary_indices:
                new_boundary_indices.append(j)
            j = j + 1
        nearby_pts = log_data[sorted_dist_indices[:j]]
        neary_pts_center = np.average(nearby_pts, axis=0)

        # # calculate svd/pca of nearby_pts in full three dimensional space and then project into two dimensions
        u, s, vT = np.linalg.svd(nearby_pts - neary_pts_center, full_matrices=False)
        planar_projection = np.dot(nearby_pts - neary_pts_center, vT[:2].T)
        planar_boundary_pts = planar_projection[new_boundary_indices]

        # # calculate svd/pca of boundary points in two dimensions and project normal to the boundary
        m, e, pT = np.linalg.svd(planar_boundary_pts - np.average(planar_boundary_pts, 0), full_matrices=False)
        # make sure we're projecting away from points
        direction = -1
        if np.dot(pT[1], planar_boundary_pts[0] - np.average(planar_projection, axis=0)) > 0:
            direction = 1
        new_projected_point = direction*step_size*pT[1] + planar_projection[0]

        # # reconstruct in three dimensions
        expanded_boundary_pts[i] = new_projected_point[0]*vT[0] + new_projected_point[1]*vT[1] + neary_pts_center

    boundary = log_data[boundary_indices]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(log_data[:,0], log_data[:,1], log_data[:,2], c='b')
    ax.scatter(boundary[:,0], boundary[:,1], boundary[:,2], c='r', s=100)
    ax.scatter(expanded_boundary_pts[:,0], expanded_boundary_pts[:,1], expanded_boundary_pts[:,2], c='g', s=100)
    ax.set_xlabel('\n\n' + r'$k_1$')
    ax.set_ylabel('\n\n' + r'$k_{-1}$')
    ax.set_zlabel('\n\n' + r'$k_2$')
    plt.locator_params(nbins=4)
    plt.show()

if __name__ == "__main__":
    rawlings_test()

    
    
