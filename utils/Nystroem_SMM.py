import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

from functools import partial

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

from sklearn.kernel_approximation import Nystroem

def pairwise_kernels_2(X, Y, metric, sigma):
    
    return metric(X, Y, sigma)

class Nystroem_SMM(Nystroem):
    
    def __init__(self, kernel="rbf", gamma=None, coef0=None, degree=None,
                 kernel_params=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Training data.
        """
#         X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        basis_kernel = pairwise_kernels_2(basis, basis, self.kernel, self.kernel_params)

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self

    def transform(self, X):
        """Apply feature map to X.
        Computes an approximate feature map using the kernel
        between some training points and X.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Data to transform.
        Returns
        -------
        X_transformed : array, shape=(n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, 'components_')
#         X = check_array(X, accept_sparse='csr')

        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels_2(X, self.components_,
                                    self.kernel,
                                    self.kernel_params)
        return np.dot(embedded, self.normalization_.T)