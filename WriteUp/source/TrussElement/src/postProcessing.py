"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, and Prof. J.P. Conte 
affiliation: University of California, San Diego

"""

#import standard python modules
import numpy as np
import random

#%%

results_output  = np.loadtxt('data.out') 

noise = []

for i in range(len(results_output)):
    noise.append(random.gauss(0,0.05))


results_output = results_output+noise

data = results_output[np.newaxis,:]

sig = np.array([0.05])

def log_likelihood(ParticleNum,par, variables, resultsLocation):
    """ this function computes the log-likelihood for parameter value: par """
     
    # num of data point: N, num of measurement channels: Ny
    N = data.shape[1]
    Ny = data.shape[0]
    
    ParameterName = variables['names']
    
    # FE predicted response
    runFEM(ParticleNum,par[:len(ParameterName)], variables, resultsLocation)
    FEpred = np.loadtxt('results.out')
    FEpred = FEpred[np.newaxis,:]
    
    
    E = np.diag(sig**2)
    logdetE = np.log(np.linalg.det(E))
    Einv = np.linalg.inv(E)
    
    # log-likelihood
    LL = -0.5*N*Ny*np.log(2*np.pi) -0.5*N*logdetE +sum(np.diag(-0.5* (data-FEpred).T @ Einv @ (data-FEpred)))
    
    return LL
