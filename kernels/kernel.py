import  numpy as np

def kernel(X,Z, gamma):
    dists_2 = np.sum(np.square(X)[:,np.newaxis,:],axis=2)-2*X.dot(Z.T)+np.sum(np.square(Z)[:,np.newaxis,:],axis=2).T
    k_XZ = np.exp(-gamma*dists_2)
    return k_XZ
