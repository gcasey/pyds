"""tesdlds.py
"""

import cv2
import sys
import numpy as np
import SimpleITK as sitk
from core import message as msg
from core.lds import lds


Y = np.genfromtxt(sys.argv[1])

try:
    dt = lds(nStates=5, doDebug=True)
    #dt.estimate(Y)
    dt.synthesize(50)
except Exception as e:
    msg.fail('Caught exception: %s' % e)
    sys.exit(-1)
