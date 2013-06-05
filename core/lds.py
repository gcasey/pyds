"""lds.py
"""

import time
import message
import numpy as np
from termcolor import colored


class lds:
    """Implements a linear dynamical system (LDS) of the form
    
    x_{t+1} = A*x_t + w_t
    y_{t}   = C*x_t + v_t
    
    where x_t is the state at time t, y_t is the observations at time t. w_t
    and v_t are the state / observation noise, respectively.
    """
    
    def __init__(self, nStates=1, doDebug=False):
        self._A = []
        self._C = []
        self._R = [] 
        self._Q = []
        self._doDebug = doDebug
        self._nStates = nStates
        
        self.__estimated = False
    
    
    def isEstimated(self): 
        return self.__estimated


    def estimate(self, Y, method='suboptimal'):
        """System identification.
    
        tbd.
        """
        try:
            { 
                "suboptimal" : self._suboptimal 
            }[method](Y)
        except KeyError:
            message.fail("%s not supported!" % method)
            return
    
    
    def synthesize(self, tau=50):
        """Synthesize obervations.
        
        tbd.
        """
        if not self.__estimated:
            raise Exception("LDS parameters not valid (for synthesis)!")
        
        Qhat = self._Qhat
        Ahat = self._Ahat
        Chat = self._Chat
        Yavg = self._Yavg
        initM0 = self._initM0
        initS0 = self._initS0
        
        
        
    
    
    def _suboptimal(self, Y):
        """Suboptimal system identification using SVD.
        
        tbd.
        """
        
        message.info("using suboptimal SVD-based estimation!")
        
        (N, tau) = Y.shape
        Yavg = np.mean(Y, axis=1)
        
        Y = Y - Yavg[:,np.newaxis]
        
        t0 = time.clock()
        (U, S, V) = np.linalg.svd(Y, full_matrices=1)
        t1 = time.clock()
        
        if self._doDebug: 
            message.info('time(SVD): %.2g [sec]' % (t1-t0))
                
        Chat = U[:,0:self._nStates]
        Xhat = (np.diag(S)[0:self._nStates,0:self._nStates] * 
                np.asmatrix(V[0:self._nStates,:]))
    
        # initial condition N(initM0, initS0)
        initM0 = np.mean(Xhat[:,0],axis=1)
        initS0 = np.zeros((self._nStates,1))
                
        pind = range(tau-1);

        phi1 = Xhat[:,pind]
        phi2 = Xhat[:,[i+1 for i in pind]]
        
        Ahat = phi2*np.linalg.pinv(phi1)
        Vhat = phi2-Ahat*phi1;
        Qhat = Vhat*Vhat.T
        
        errorY = Y - Chat*Xhat
        Rhat = np.var(errorY.ravel())
        
        self._initS0 = initS0
        self._initM0 = initM0
        self._Yavg = Yavg
        self._Ahat = Ahat
        self._Chat = Chat
        self._Xhat = Xhat
        self._Qhat = Qhat
        self._Rhat = Rhat
        
        self.__estimated = True
        
        