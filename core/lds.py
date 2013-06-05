"""lds.py
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import time
import message
import numpy as np
from termcolor import colored
from sklearn.utils.extmath import randomized_svd

class lds:
    """Implements a linear dynamical system (LDS) of the form:
    
    x_{t+1} = A*x_t + w_t
    y_{t}   = C*x_t + v_t
    
    where x_t is the state at time t, y_t is the observations at time t. w_t
    and v_t are the state / observation noise, respectively.
    """
    
    def __init__(self, nStates=1, doDebug=False):
        self._Ahat = None
        self._Chat = None
        self._Rhat = None
        self._Qhat = None
        self._Bhat = None
        self._initM0 = None
        self._initS0 = None
        self._doDebug = doDebug
        self._nStates = nStates
        
        self._estimated = False
        
    
    def synthesize(self, tau=50, mode=None):
        """Synthesize obervations.
        
        Parameters
        ----------
        
        tau : int (default = 50)
            Synthesize tau frames. 
            
        mode : Combination of ['s','q','r']
            's' - Use the original states
            'q' - Do NOT add state noise
            'r' - Add observations noise

            In case 's' is specified, 'tau' is ignored and the number of 
            frames equals the number of state time points.
            
        Returns
        -------
        
        I : numpy array, shape = (D, tau)
            Matrix with N D-dimensional column vectors as observations.
            
        X : numpy array, shape = (N, tau) 
            Matrix with N tau-dimensional state vectors.        
        """
        
        Bhat = None
        Xhat = self._Xhat
        Qhat = self._Qhat
        Ahat = self._Ahat
        Chat = self._Chat
        Rhat = self._Rhat
        Yavg = self._Yavg
        initM0 = self._initM0
        initS0 = self._initS0
        states = self._nStates
                    
        if mode is None:
            raise Exception("No mode specified (synthesis)")
        
        # use original states -> tau is restricted
        if mode.find('s') >= 0:
            tau = Xhat.shape[1]
            if self._doDebug:
                message.info('setting tau=%d' % tau)
        
        # data to be filled and returned     
        I = np.zeros((len(Yavg), tau))
        X = np.zeros((self._nStates, tau))
        
        if mode.find('r') >= 0:
            stdR = np.sqrt(Rhat)
        
        # add state noise, unless user explicitely decides against
        if not mode.find('q') >= 0:
            if self._doDebug: 
                message.info('adding state noise ...')
            stdS = np.sqrt(initS0)
            (U, S, V) = np.linalg.svd(Qhat, full_matrices=False)
            Bhat = U*np.diag(np.sqrt(S)) 
    
        t = 0 
        Xt = np.zeros((self._nStates,1))
        while (tau<0) or (t<tau):  
            # uses the original states
            if mode.find('s') >= 0:
                Xt1 = Xhat[:,t]
            # first state
            elif t == 0:
                Xt1 = initM0;
                if mode.find('q') < 0:
                    Xt1 += stdS*np.rand(self._nStates)
            # any further states (if mode != 's')
            else:
                Xt1 = Ahat*Xt
                if not mode.find('q') >= 0:
                    Xt1 = Xt1 + Bhat*np.rand(self._nStates)
            
            # synthesizes image
            It = Chat*Xt1 + np.reshape(Yavg,(len(Yavg),1))
         
            # adds observation noise
            if mode.find('r') >= 0:
                It += stdR*np.randn(length(Yavg))
            
            # save ...
            Xt = Xt1;
            I[:,t] = It.reshape(-1)
            X[:,t] = Xt.reshape(-1)
            t += 1
            
        return (I, X)
    
        
    def suboptimalSysID(self, Y, approximate=False):
        """Suboptimal system identification using SVD.
        
        Suboptimal system identification based on SVD, as proposed in the 
        original work of Doretto et al. [1].
    
        All the interal LDS parameters are updated.
        
        Parameters
        ----------
        
        Y : numpy array, shape = (N, D)
            Input data with D observations as N-dimensional column vectors.

        approximate : True|False
            Use randomized (fast) SVD computation.
                
        Returns
        -------
        """
        
        message.info("using suboptimal SVD-based estimation!")
        
        (N, tau) = Y.shape
        Yavg = np.mean(Y, axis=1)
        
        Y = Y - Yavg[:,np.newaxis]
        
        t0 = time.clock()
        if approximate:
            (U, S, V) = randomized_svd(Y, self._nStates)
        else:
            (U, S, V) = np.linalg.svd(Y, full_matrices=0)
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
        Qhat = 1.0/Vhat.shape[1] * Vhat*Vhat.T 
         
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
        
        