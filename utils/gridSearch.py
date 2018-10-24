import numpy as np
from operator import itemgetter
from models.OCSMM import OCSMM

def empirical_p_values(scores):
    n_sets = scores.shape[0]
    scores = np.squeeze(scores.tolist())
    L = [[idx, score] for idx, score in zip(range(n_sets), scores)]
    L = sorted(L, key=itemgetter(1))
    ecdf = np.arange(1, n_sets+1)/n_sets
    empirical_p = np.array([(1.-ecdf[key[0]]) for key in L])
    return empirical_p

def roc_auc(Ytest, emp_p_values):
    """
    Input
    ----------
    Ytest :list-like, list of labeled data 'normal'
    grids : grids with hyperparameters to tune (gammas, Cs)
    Output
    ----------
    performance: dictionary with (hyperparameters): (AUC, Accuracy)
    """
    positives = Ytest
    negatives = [1-y for y in Ytest]
    cuts = np.arange(len(Ytest))*(1/len(Ytest))
    
    best_cut = 0.
    tprs, fprs = [], []
    for cut in cuts:
        tpr = sum([a*b for a,b in zip(emp_p_values < cut, positives)])/sum(positives)
        fpr = sum([a*b for a,b in zip(emp_p_values < cut, negatives)])/sum(negatives)
        if tpr==1. and best_cut==0.:
            best_cut=cut
        tprs.append(tpr); fprs.append(fpr)
    lag_tprs = [0]+tprs[0:len(tprs)-1]
    diff_tprs = [tpr-lag_tpr for tpr,lag_tpr in zip(tprs, lag_tprs)]
    auc = 1.-sum([fpr*diff_tpr for fpr,diff_tpr in zip(fprs, diff_tprs)])
    roc_curve = np.array([fprs, tprs]).T
    return auc, best_cut, roc_curve

def accuracy(Ytest, Ypred):
    return sum(Ytest==Ypred)/len(Ytest)

def gridSearch(Strain, Stest, Ytest, grids):
    """
    Input
    ----------
    Strain: list-like, list of training data matrices with shape (n_samples, n_features)
    Stest : list-like, list of test data matrices with shape (n_samples, n_features)
    Ytest : list-like, labels for data sets 'normal':0, 'anomaly':1
    grids : grids with hyperparameters to tune (gammas, Cs)
    Output
    ----------
    performance: dictionary with (C, gamma): (AUC, Accuracy)
    """
    grid_gamma, grid_C = grids['gammas'], grids['Cs']
    performance = {}
    for C in grid_C:
        for gamma in grid_gamma:
            clf = OCSMM(Strain, C, gamma)
            clf.fit()
            scores = clf.decision_function(Stest)
            emp_p_values = empirical_p_values(scores)
            auc, best_cut, _ = roc_auc(Ytest, emp_p_values)
            Ypred = (emp_p_values < best_cut)*1
            acc = accuracy(Ytest, Ypred)
            performance[(C, gamma)] = (auc, acc)
    return performance
