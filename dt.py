"""ldstool.py

Shows how to use LDS's to estimate and synthesize dynamic textures.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import os
import cv2
import sys
import numpy as np
import cv2.cv as cv
import core.dsutil as dsutil
from core.lds import lds
from core import message
from optparse import OptionParser


def usage():
    """Print usage information"""
    print("""
Dynamic Texture estimation / synthesis using Linear Dynamical Systems (LDS).

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -i ARG -- Input file
    -t ARG -- Input file type
    
        'vFile' - AVI video file
        'aFile' - ASCII data file
        'lFile' - Image list file 
        
    [-n ARG] -- LDS states (default: 5)
    [-m ARG] -- FPS for movie display (default: 20)
        
        Shows the input movie, or (if -s), the synthesis
        result.
    
    [-e] -- Run estimation (default: False)
    [-s] -- Run synthesis  (default: False)
    [-v] -- Verbose output (default: False)
    [-a] -- Use randomized SVD for estimation
    
    
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-i", dest="iFile") 
    parser.add_option("-t", dest="iType")
    parser.add_option("-n", dest="nStates", type="int", default=+5)
    parser.add_option("-m", dest="doMovie", type="int", default=-1)
    parser.add_option("-a", dest="svdRand", action="store_true", default=False)
    parser.add_option("-e", dest="doEstim", action="store_true", default=False)
    parser.add_option("-s", dest="doSynth", action="store_true", default=False)
    parser.add_option("-h", dest="shoHelp", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    opt, args = parser.parse_args()
    
    if opt.shoHelp: 
        usage()
    
    dataMat, dataSiz = None, None
    if opt.iType == 'vFile':
        (dataMat, dataSiz) = dsutil.loadDataFromVideoFile(opt.iFile)
    elif opt.iType == 'aFile':
        (dataMat, dataSiz) = dsutil.loadDataFromASCIIFile(opt.iFile)
    elif opt.iType == 'lFile':
        (dataMat, dataSiz) = dsutil.loadDataFromIListFile(opt.iFile)
    else:
        message.fail("Unsupported file type : %s", opt.iType)    
        sys.exit(-1)
    
    dt = lds(nStates=opt.nStates, 
             verbose=opt.verbose)

    if opt.doEstim:
        try:
            dt.suboptimalSysID(dataMat, approximate=opt.svdRand)
        except Exception as e:
            message.fail("Oops ... : %s" % e)
            sys.exit(-1)
    
    dataMatSyn, dataMatStates = None, None
    if opt.doSynth:
        try:
            (dataMatSyn, dataMatStates) = dt.synthesize(tau=50, mode='s')
        except Exception as e:
            message.fail("Oops ... : %s" % e)
            sys.exit(-1)
                
    if opt.doMovie > 0:
        if dataMatSyn is None:
            dsutil.showMovie(frames=dataMat, movSz=dataSiz, fps=opt.doMovie)
        else:
            dsutil.showMovie(frames=dataMatSyn, movSz=dataSiz, fps=opt.doMovie)    
                    
                    
if __name__ == '__main__':
    sys.exit(main())