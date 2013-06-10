"""ldsdist.py
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import time
import pickle
import message
import numpy as np
import scipy.linalg
from termcolor import colored
from sklearn.utils.extmath import randomized_svd


def ldsMartinDistance(lds1, lds2, N=20):
    """Martin distance between two LDS's.
    
    Algorithmic outline:
    --------------------
    
    This code implements the subspace angle approach of
    
    [1] K. De Cock and B. De Moor, "Subspace angles between ARMA models", In:
        Systen & Control Letters, vol 46, pp. 265-270, 2002

    that is used in many works which implement the Martin distance as a 
    similarity measure linear dynamical systems.
    
    Parameters:
    -----------
    lds1: core.lds instance
        First LDS model.
    
    lds2: core.lds instance
        Second LDS model.
    
    N : int (default: 20)
        Number of iterations to compute the "infinite sum" that is the 
        solution to the Lyapunov equation (see code.)
    
    Returns:
    --------
    D : np.float32
        Martin distance between lds1 and lds2.
    """
    
    if not lds1.check() or not lds2.check():
        message.fail("Models are incomplete!")
        raise Exception()
    
    # get relevant params
    C1 = lds1._params["Chat"]
    C2 = lds2._params["Chat"]
    A1 = lds1._params["Ahat"]
    A2 = lds2._params["Ahat"]
    
    C1C1 = np.asmatrix(C1).T*C1
    C2C2 = np.asmatrix(C2).T*C2
    C1C2 = np.asmatrix(C1).T*C2
    
    dx1 = len(lds1._params["initM0"])
    dx2 = len(lds2._params["initM0"])
    
    # matrices that are used for the GEP
    K = np.zeros((dx1+dx2, dx1+dx2))
    L = np.zeros((dx1+dx2, dx1+dx2))
    
    # N summation terms
    for i in range(N+1):
        if i == 0:
            O1O2 = C1C2
            O1O1 = C1C1
            O2O2 = C2C2
            a1t = A1
            a2t = A2
        else:
            O1O2 = O1O2 + a1t.T*C1C2*a2t
            O1O1 = O1O1 + a1t.T*C1C1*a1t
            O2O2 = O2O2 + a2t.T*C2C2*a2t
            if i != N-1:
                a1t = a1t*A1
                a2t = a2t*A2
                
        # we are at the end
        if i == N-1:
            K[0:dx1,dx1:] = O1O2
            K[dx1:,0:dx1] = O1O2.T
            L[0:dx1,0:dx1] = O1O1
            L[dx1:,dx1:] = O2O2
            ev = np.flipud(np.sort(np.real(scipy.linalg.eigvals(K,L))))
            if len(np.nonzero(ev)[0]) != len(ev):
                return np.inf
            else:
                return -2*np.sum(np.log(ev[0:dx1]))
            
            
            
    
    