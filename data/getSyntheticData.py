import numpy as np

def genSyntheticSet(n, pmix=None, seed=1):
    # Generate 2D Gaussian Mixture sets using pmix weights or 'normal' weights.
    np.random.seed(seed=seed)    
    # Topic sampling
    ws = [[0.33, 0.33, 0.33],[0.84, 0.08, 0.08]]
    if pmix==None:
        pmix = ws[np.random.binomial(1, p=.5)]
    n_topics = np.random.multinomial(n, pvals=pmix)
    # Mixture sampling
    mus = [[-1.7, -1], [1.7, -1], [0,2]]
    sigma = 0.2 * np.identity(2)
    X = []
    for idx, n_topic in enumerate(n_topics):
        if n_topic>0:
            X.append(np.random.multivariate_normal(mean = mus[idx], cov = sigma, size=n_topic))
    X = np.vstack(X)
    return X

def genAnomalousSyntheticSet(n, pmix=None, seed=2):
    # Random choose between 2 anomalous different Gaussian Mixtures sets or 1 anomalous Gaussian set.
    # Generate n samples.
    np.random.seed(seed=seed)
    idx = np.argmax(np.random.multinomial(1, pmix, size=1))
    if idx==0:
        X_anomaly = genSyntheticSet(n, pmix=[0.84, 0.08, 0.08])
        #X_anomaly = np.random.multivariate_normal(mean = [0, 0], cov = np.identity(2), size=n)
    elif idx==1:
        X_anomaly= genSyntheticSet(n, pmix=[0.33, 0.64, 0.03])
        #X_anomaly = np.random.multivariate_normal(mean = [0, 0], cov = np.identity(2), size=n)
    else:
        X_anomaly = np.random.multivariate_normal(mean = [0, 0], cov = np.identity(2), size=n)
    return X_anomaly

def genSyntheticTrainData(n_sample, n_sets):
    # Generate Train data composed of n_sets 2D Gaussian Mixtures.
    Xtrain = []
    n_samples = np.random.poisson(lam=n_sample, size=n_sets).tolist()
    Xtrain = [genSyntheticSet(n, seed=s) for s,n in enumerate(n_samples)]
    return Xtrain

def getSyntheticTestData(n_sample, n_sets):
    # Generate 20 normal data sets and 10 anomalous.
    # Labels 'normal':0, 'anomaly':1
    Ytest = [0]*20+[1]*10
    Xtest = []
    n_samples = np.random.poisson(lam=n_sample, size=n_sets).tolist()
    for s,y in enumerate(Ytest):
        if y==0: 
            Xtest.append( genSyntheticSet(n_samples[s], seed=(s+100)) )
        elif y==1: 
            Xtest.append( genAnomalousSyntheticSet(n_samples[s], pmix=[.33, .33, .33], seed=(s+100)) )
    return Xtest, Ytest
