import numpy as np

center1 = np.array([0,0])
center2 = np.array([4,3])

nbsamples = 1000

bluepoints = np.random.multivariate_normal(mean=center2,cov=np.identity(2),size=1000)
fbp= np.c_[bluepoints,np.ones(nbsamples)]

redpoints = np.random.multivariate_normal(mean=center1,cov=np.identity(2),size=1000)
frp = np.c_[bluepoints,-np.ones(nbsamples)]

rData = np.concatenate((fbp, frp), axis=0)

mData = rData[np.random.permutation(2*nbsamples)]

print(1)
