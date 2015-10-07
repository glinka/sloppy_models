data = np.genfromtxt('./data/params_and_of_evals.csv', delimiter=',')
of_tol = 0.4 # from plotting with scratch.py
somedata = data[data[:,0] < of_tol]
# only keep npts points due to computational considerations
npts = 6000
slice_size = somedata.shape[0]/npts
somedata = somedata[::slice_size]
log_params_data = np.log10(somedata[:,1:])
# add some noise
noise_level = 0.02
log_params_data = log_params_data + noise_level*np.random.normal(size=log_params_data.shape)
keff = somedata[:,1]*somedata[:,3]/(somedata[:,2] + somedata[:,3])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# cannot use log axes in 3d plot, so plot log(data) directly
ax.scatter(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2])
ax.set_xlabel('log(k1)')
ax.set_ylabel('log(kinv)')
ax.set_zlabel('log(k2)')
# ax.set_xlim((np.min(somedata[:,1]),np.max(somedata[:,1])))
# ax.set_ylim((np.min(somedata[:,2]),np.max(somedata[:,2])))
# ax.set_zlim((np.min(somedata[:,3]),np.max(somedata[:,3])))
# ax.xaxis.set_scale('log')
# ax.yaxis.set_scale('log')
# ax.zaxis.set_scale('log')

# dmaps stuff
import plot_dmaps

eigvects = np.genfromtxt('./data/dmaps-eigvects--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', delimiter=',')
eigvals = np.genfromtxt('./data/dmaps-eigvals--tol-' + str(of_tol) + '-k-' + str(k) + '.csv', delimiter=',')
plt.scatter(eigvects[:,1], np.ones(eigvects.shape[0]), c=np.log10(keff), lw=0)
plt.show()

plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,1], xlabel='log(k1)', ylabel='log(kinv)', zlabel='log(k2)')
plot_dmaps.plot_xyz(log_params_data[:,0], log_params_data[:,1], log_params_data[:,2], color=eigvects[:,2], xlabel='log(k1)', ylabel='log(kinv)', zlabel='log(k2)')
