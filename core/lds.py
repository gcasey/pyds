"""lds.py
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import copy
import time
import pickle
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
    
    def __init__(self, verbose=False):
        """Initialize instance.
        """
        
        self._params = {
            "Ahat" : None,
            "Chat" : None,
            "Rhat" : None,
            "Qhat" : None,
            "Xhat" : None,
            "Yavg" : None,
            "initM0" : None,
            "initS0" : None,
            "nStates" : None
        }
        
        # LDS instance not ready at that point
        self._ready = False
        self._verbose = verbose

    
    def check(self):
        """Check validity of LDS parameters.
     
        Currently, this routine only checks if the parameters are set, but not
        if they are actually valid parameters!
        
        Returns:
        --------
        validity : boolean
            True if parameters are valid, False otherwise.
        """
        
        for key in self._params.keys():
           if self._params[key] is None: return False
        return True
      
        
    def save(self, outFile):
        """Pickle data to disk.
        
        Parameters
        ----------
        outFile : Filename of output file
        """
        
        if not self.check():
            message.fail("Cannot write non-ready LDS!")
            raise Exception()
        
        pickle.dump(self._params, open(outFile, 'w'))
        
    
    def load(self, inFile): 
        """Load pickled parameters from disk.
        
        Parameters
        ----------
        inFile : string
            Filename of input file.
        """
        
        if self._verbose:
            message.info("loading LDS parameters ...")
        self._params = pickle.load(open(inFile))
        
        if self.check():
            self._ready = True
        else:
            message.warn("Loaded model not ready!")
            
    
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
        
        if not self._ready:
            message.fail("LDS not ready for synthesis!")
            raise Exception()
        
        Bhat = None
        Xhat = self._params["Xhat"]
        Qhat = self._params["Qhat"]
        Ahat = self._params["Ahat"]
        Chat = self._params["Chat"]
        Rhat = self._params["Rhat"]
        Yavg = self._params["Yavg"]
        initM0 = self._params["initM0"]
        initS0 = self._params["initS0"]
        nStates = self._params["nStates"]
        
        if mode is None:
            message.fail("No synthesis mode specified!")
            raise Exception()
        
        # use original states -> tau is restricted
        if mode.find('s') >= 0:
            tau = Xhat.shape[1]
        
        # data to be filled and returned     
        I = np.zeros((len(Yavg), tau))
        X = np.zeros((nStates, tau))
        
        if mode.find('r') >= 0:
            stdR = np.sqrt(Rhat)
        
        # add state noise, unless user explicitly decides against
        if not mode.find('q') >= 0:
            stdS = np.sqrt(initS0)
            (U, S, V) = np.linalg.svd(Qhat, full_matrices=False)
            Bhat = U*np.diag(np.sqrt(S)) 
    
        t = 0 
        Xt = np.zeros((nStates, 1))
        while (tau<0) or (t<tau):  
            # uses the original states
            if mode.find('s') >= 0:
                Xt1 = Xhat[:,t]
            # first state
            elif t == 0:
                Xt1 = initM0;
                if mode.find('q') < 0:
                    Xt1 += stdS*np.rand(nStates)
            # any further states (if mode != 's')
            else:
                Xt1 = Ahat*Xt
                if not mode.find('q') >= 0:
                    Xt1 = Xt1 + Bhat*np.rand(nStates)
            
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
    
        
    def suboptimalSysID(self, Y, nStates, approximate=False):
        """Suboptimal system identification using SVD.
        
        Suboptimal system identification based on SVD, as proposed in the 
        original work of Doretto et al. [1]. 
        
        Parameters
        ----------
        Y : numpy array, shape = (N, D)
            Input data with D observations as N-dimensional column vectors.

        nStates : int
            Number of LDS states to estimate.

        approximate : True|False (optional, default: False)
            Use randomized (fast) SVD computation.
        """
        
        if self._verbose:
            message.info("using suboptimal SVD-based estimation!")

        if nStates < 0:
            message.fail("#states < 0!")
            raise Exception()

        (N, tau) = Y.shape
        Yavg = np.mean(Y, axis=1)
        Y = Y - Yavg[:,np.newaxis]
        
        t0 = time.clock()
        if approximate:
            (U, S, V) = randomized_svd(Y, nStates)
        else:
            (U, S, V) = np.linalg.svd(Y, full_matrices=0)
        t1 = time.clock()
                
        if self._verbose: 
            message.info('time(SVD): %.2g [sec]' % (t1-t0))
                
        Chat = U[:,0:nStates]
        Xhat = (np.diag(S)[0:nStates,0:nStates] * np.asmatrix(V[0:nStates,:]))
    
        # initial condition N(initM0, initS0)
        initM0 = np.mean(Xhat[:,0], axis=1)
        initS0 = np.zeros((nStates, 1))
                
        pind = range(tau-1);

        phi1 = Xhat[:,pind]
        phi2 = Xhat[:,[i+1 for i in pind]]
        
        Ahat = phi2*np.linalg.pinv(phi1)
        Vhat = phi2-Ahat*phi1;
        Qhat = 1.0/Vhat.shape[1] * Vhat*Vhat.T 
         
        errorY = Y - Chat*Xhat
        Rhat = np.var(errorY.ravel())
        
        # save parameters
        self._params["nStates"] = nStates
        self._params["initS0"] = initS0
        self._params["initM0"] = initM0
        self._params["Yavg"] = Yavg
        self._params["Ahat"] = Ahat
        self._params["Chat"] = Chat
        self._params["Xhat"] = Xhat
        self._params["Qhat"] = Qhat
        self._params["Rhat"] = Rhat
        
        # the LDS is ready
        self._ready = True
 
 
    @staticmethod
    def stateSpaceMap(lds1, lds2):
        """
        Map parameters from lds1 into space of lds2 (state-space).
        
        Parameters:
        -----------
        lds1 : lds instance
            Target LDS
        
        lds2: lds instance
            Source LDS
        
        Returns:
        --------
        lds : lds instance
            New instance of lds2 (with UPDADED parameters)
            
        err : float
            Absolute difference between the vectorized parameter sets before
            the state-space mapping.
        """
        
        # make a shallow copy (no compound object -> no problem)
        lds = copy.copy(lds2)

        Chat1 = lds1._params["Chat"]
        Chat2 = lds2._params["Chat"]
        
       
        F = np.asmatrix(np.linalg.pinv(Chat2))*Chat1
     
        # compute TRANSFORMED params (rest should be kept the same)
        lds._params["Chat"] = lds2._params["Chat"]*F
        lds._params["Ahat"] = F.T*lds2._params["Ahat"]*F
        lds._params["Qhat"] = F.T*lds2._params["Qhat"]*F
        lds._params["Rhat"] = lds2._params["Rhat"]
        lds._params["initM0"] = F.T*lds2._params["initM0"]
        lds._params["initS0"] = np.diag(F.T*np.diag(lds._params["initS0"].ravel())*F)
        
        err = 0
        err += np.sum(np.abs(lds2._params["Chat"].ravel() - 
                             lds1._params["Chat"].ravel()))
        err += np.sum(np.abs(lds2._params["Ahat"].ravel() - 
                             lds1._params["Ahat"].ravel()))
        err += np.sum(np.abs(lds2._params["Qhat"].ravel() - 
                             lds1._params["Qhat"].ravel()))
        err += np.sum(np.abs(lds2._params["Rhat"].ravel() - 
                             lds1._params["Rhat"].ravel()))
        err += np.sum(np.abs(lds2._params["initM0"].ravel() - 
                             lds1._params["initM0"].ravel()))                        
        err += np.sum(np.abs(lds2._params["initS0"].ravel() - 
                             lds1._params["initS0"].ravel()))                        
        err += np.sum(np.abs(lds2._params["Yavg"].ravel() - 
                             lds1._params["Yavg"].ravel()))                        
        return (lds, err)
