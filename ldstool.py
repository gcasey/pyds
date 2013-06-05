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
from core.lds import lds
from core import message
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


def loadVideo(videoFile):
    """Load AVI video and convert to observation matrix.
    
    Tries to load an AVI video file, converts all frames to grayscale (in 
    case of color frames) and returns the video in the form of a matrix 
    where each frame is a column vector.
    
    Parameters:
    -----------
    
    videoFile : string
        Name of the AVI input video file (might be color).
        
    Returns:
    --------
    
    Y : numpy array, shape ((H*W),F)
        Output matrix with F (H*W)-dimensional column vectors, where H and W
        are the frame height and width, respectively.
    """
    
    # tries to open AVI video file
    capture = cv2.VideoCapture(videoFile)

    # gets the number of frames in the video file
    nFrames = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

    flag, frame = capture.read()
    if flag == 0:
        raise Exception("Oops ...")
    
    # reads first frame, converts frame to grayscale and gets size
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nPix = frame.shape[0]*frame.shape[1]

    # create observation matrix
    Y = np.zeros((nPix, nFrames), dtype=np.float32)
    Y[:,0] = frame.reshape(-1)
    
    # reads the rest ...
    cnt = 1
    while True:
        flag, frame = capture.read()
        if flag == 0:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Y[:,cnt] = frame.reshape(-1) 
        cnt += 1
        
    return Y
    
def main(argv=None):
    if argv is None: argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-i", dest="iVideo") 
    parser.add_option("-c", dest="ldsMod")
    parser.add_option("-s", dest="oVideo")
    parser.add_option("-h", dest="doHelp", action="store_true", default=False)
    options, args = parser.parse_args()
    
    if options.doHelp:
        usage()
        sys.exit(-1)

    iVideo = options.iVideo
    ldsMod = options.ldsMod
    oVideo = options.oVideo
    
    if not oVideo is None:
        if ldsMod is None:
            message.fail('LDS model required (see -h)')
            sys.exit(-1)
        if not os.path.exists(ldsMod):
            message.fail('LDS model %s not found!' % lsdMod)
            sys.exit(-1)

    if not os.path.exists(iVideo):
        message.fail('Video file %s not found!' % iVideo)
        sys.exit(-1)

    Y = loadVideo(iVideo)

    #Y = np.genfromtxt(sys.argv[1], dtype=np.float32)

    dt = lds(nStates=5, doDebug=True)
    dt.suboptimalSysID(Y)
    #(I, X) = dt.synthesize(50, mode='s')


if __name__ == '__main__':
    sys.exit(main())