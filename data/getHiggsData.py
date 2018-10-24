import os
import numpy as np
from numpythia import Pythia, hepmc_write

def get_group_events(name, num_events, seed, train):
    # Generate group of Monte Carlo simulated proton-antiproton collision events 
    # at the LHC, composed by particle readings (px,py,pz,e,m).
    # Input: name.cmnd config file for Pythia simulator.
    # generate events while writing to ascii hepmc
    momentum_four = ['px', 'py', 'pz', 'E'] #'mass', 'pdgid'
    group_events = []
    pythia = Pythia(os.path.join('./data/cfgs/', name + '.cmnd'), random_state=seed)
    for event in hepmc_write('events.hepmc', pythia(events=num_events)):
        particles = event.all()
        if train==True:
            # If training, then return only background signals, not Higgs reading.
            particles = particles[particles['pdgid']!=25]
        # Filter and transform np structured array to ndarray
        X = particles[momentum_four].copy()
        particles = X.view(np.float64).reshape(X.shape + (-1,))
        group_events.append(particles)
    return group_events

def genTrainSet(cfg, num_events, seed=1):
    X = get_group_events(cfg, num_events, seed, train=True)
    return X

def genTestSet(cfg, events, seed=101):
    # Events, list with number of background samples and Higgs realization.
    Ytest = [0]*events[0]+[1]*events[1]
    X = get_group_events(cfg, num_events=events[0], seed=seed, train=True)
    X_Higgs = get_group_events(cfg, num_events=events[1], seed=seed, train=False)
    X.extend(X_Higgs)
    return X, Ytest
