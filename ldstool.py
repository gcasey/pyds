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
from core.lds import lds
from core import message as msg
from optparse import OptionParser


def usage():
    """Print usage information"""
    print("""
Estimates / Synthesizes a dynamic texture from an AVI video file.

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -i FILE 
    -m FILE
    -s FILE

OPTIONS (Detailed):

    -i FILE
    
    FILE is the name of the input AVI file.
  
    -m FILE
    
    FILE is the name of the output / input model file.
  
    -s FILE
    
    FILE is the name of the synthesized video (requires -m)
  
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))


if __name__ == "__main__":

    parser = OptionParser(add_help_option=False)
    parser.add_option("-i", dest="iVideo") 
    parser.add_option("-c", dest="ldsMod")
    parser.add_option("-s", dest="oVideo")
    parser.add_option("-h", dest="doHelp", action="store_true", default=False)
    options, args = parser.parse_args()
    
    if options.doHelp:
        usage()
        sys.exit(-1)

    Y = np.genfromtxt(sys.argv[1], dtype=np.float32)

    #dt = lds(nStates=5, doDebug=True)
    #dt.suboptimalSysID(Y)
    #(I, X) = dt.synthesize(50, mode='s')


if __name__ == '__main__':
    sys.exit(main())