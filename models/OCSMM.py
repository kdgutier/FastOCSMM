import numpy as np
from kernels.kernel import kernel
from kernels.gramMeasure import gramMeasure, cross_gramMeasure
import cvxopt

class OCSMM():
    """One Class Support Measure Machines (OCSMM).
    Implementation of algorithm by Muandet & Scholkopf

    Input
    ----------
    S : list-like, list of training data matrices with shape (n_samples, n_features)
    C : regularization parameter

    References
    ----------
    .. [MuandetScholkpf2013] One-Class Support Measure Machines for 
        Group Anomaly Detection, 2013, Proceedings of the Twenty-Ninth 
        Conference on Uncertainty in Artificial Intelligence
    """
    def __init__(self, Strain, C, gamma):
        self.C = C
        self.Strain = Strain
        self.gamma = gamma

    def fit(self):
        # Useful Variables
        n_sets = len(self.Strain)
        self.Gram_train = gramMeasure(self.Strain, self.gamma)
        ones = np.ones(shape=(n_sets,1))
        zeros = np.zeros(shape=(n_sets,1))
        # Construct Quadratic Problem matrices for cvxopt P, q, G, h, A, b
        P = cvxopt.matrix(self.Gram_train)
        q = cvxopt.matrix(zeros)
        G = cvxopt.matrix(np.vstack((-np.identity(n_sets), np.identity(n_sets))))
        h = cvxopt.matrix(np.vstack((zeros, self.C*ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)
        # Solve (min QP) with CVXOPT 
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        self.alpha =np.ravel(solution['x'])
        self.dual_objective = np.ravel(solution['primal objective'])
    
    def decision_function(self, Stest):
        self.Gram_test = gramMeasure(Stest, self.gamma)
        self.Gram_cross = cross_gramMeasure(self.Strain, Stest, self.gamma)
        self.sms_idxs = np.squeeze(np.where(self.alpha > 1e-5))
        ones = np.ones(shape=(len(Stest), 1))
        cross_prod = np.matmul(self.Gram_cross[self.sms_idxs,:].T, np.expand_dims(self.alpha[self.sms_idxs],axis=1))
        ypred = ones -(2*cross_prod)+self.dual_objective*ones
        return ypred
        