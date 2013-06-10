"""dtdist.py
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
from core.ldsdist import ldsMartinDistance
import core.dsutil as dsutil
from core.lds import lds
from core import message
from optparse import OptionParser


def usage():
    """Print usage information"""
    print("""
Distance computation between Dynamic Texture (DT) models.

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -s ARG -- DT1 model file
    -r ARG -- DT2 model file
    -t ARG -- Distance
    
        Supported distances are:
        
            'Martin' -- Martin distance
        
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-s", dest="model1File")
    parser.add_option("-r", dest="model2File") 
    parser.add_option("-n", dest="iterations", type="int", default=20)
    parser.add_option("-h", dest="shoHelp", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    opt, args = parser.parse_args()
    
    if opt.shoHelp: 
        usage()
    
    # create instances of two LDS's
    dt1 = lds(verbose=opt.verbose)
    dt2 = lds(verbose=opt.verbose)
    
    # load parameters
    dt1.load(opt.model1File)
    dt2.load(opt.model2File)
    
    print ldsMartinDistance(dt1, dt2, opt.iterations)
    (dt2Prime, err) = lds.stateSpaceMap(dt1, dt2)
    print ldsMartinDistance(dt1, dt2Prime, opt.iterations)
    
    
    
        
        
        
            
if __name__ == '__main__':
    sys.exit(main())