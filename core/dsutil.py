"""dsutil.py

I/O and utility routines for dynamical systems.
"""

import cv2
import cv2.cv as cv
import numpy as np
import message as message


def renormalize(data, (newmin, newmax), oldrange=None):
    """Normalize data into requested range.
    
    Parameters
    ----------
    data : numpy array, shape = (N, D)
        Input data.
        
    (newmin, newmax) : tuple of min. and max. value
        The range we want the values to be in.
    
    oldrange : tupe of user-specified input range
        Input range to use (Note: no clipping is done)
    
    Returns
    -------
    out : numpy array, shape = (N, D)
        Scaled output data.
    """
    
    data = data.astype('float32')
    if oldrange is None:
        (oldmin, oldmax) = (np.min(data), np.max(data))
    else:
        (oldmin, oldmax) = oldrange
    slope = (newmin-newmax+0.)/(oldmin-oldmax)
    out = slope*(data-oldmin) + newmin
    return out    
    
    
def showMovie(frames, movSz, fps=20):
    """Show a movie using OpenCV.
    
    Takes a numpy matrix (with images as columns) and shows the images in 
    a video at a specified frame rate.
    
    Parameters
    ----------
    frames : numpy array, shape = (N, D)
        Input data with N images as D-dimensional column vectors.
        
    movSz : tupe of (height, width, nFrames)
        Size of the movie to show. This is used to reshape the column vectors
        in the data matrix.
    
    fps : int (default: 20)
        Show video at specific frames/second (FPS) rate.
    """

    if fps < 0:
        raise Exception("FPS < 0")
    
    video = frames.reshape(movSz)
    nImgs = frames.shape[1]
    tWait = int(1000.0/fps);

    for i in range(nImgs):
        cv2.imshow("video", renormalize(video[:,:,i].T, (0, 1)))
        key = cv2.waitKey(tWait)
        if key == 27: 
            break
    
    
def loadDataFromVideoFile(inFile):
    """Read an AVI video into a data matrix.
    
    Parameters
    ----------
    inFile : string
        Name of the AVI input video file (might be color - if so, it will be 
        converted to grayscale).
        
    Returns
    -------
    dataMat : numpy array, shape = (N, D)
        Output data matrix, where N is the number of pixel in each of the D
        frames.
        
    dataSiz : tuple of (height, width, D)    
        The video dimensions.
    """

    capture = cv2.VideoCapture(inFile)

    flag, frame = capture.read()
    if flag == 0:
        raise Exception("Could not read %s!" % inFile)
    
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    N = np.prod(frame.shape)
    D = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
    (height, width) = frame.shape

    dataMat = np.zeros((N, D), dtype=np.float32)
    dataMat[:,0] = frame.reshape(-1)
    
    cnt = 1
    while True:
        flag, frame = capture.read()
        if flag == 0:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dataMat[:,cnt] = frame.reshape(-1)
        cnt += 1  

    return (dataMat, (height, width, D))
    

def loadDataFromASCIIFile(inFile):
    """Read an ASCII file into a data matrix.
    
    Reads a data matrix from a file. The first line of the ASCII file needs
    to contain a header of the form (HEIGHT WIDTH FRAMES), e.g.,
    
    2 2 3
    1.2 3.4 5.5
    1.1 1.1 2.2
    4.4 1.1 2.2
    3.3 3.3 1.1
    
    The header information will be returned to the user so that the data
    matrix can later be reshaped into an actual video.
    
    Parameters
    ----------
    inFile : string
        Input file name.
    
    Returns
    -------
    dataMat : numpy array, shape = (N, D)
        Output data matrix, where N = (width x height)
        
    dataSiz : tuple of (height, width, D)    
        The video dimensions.
    """

    with open(inFile) as fid:
        header = fid.readline()
    
    dataSiz = [int(x) for x in header.split(' ')]
    if len(dataSiz) != 3:
        message.fail("Invalid header!")
        raise Exception()
        
    
    dataMat = np.genfromtxt(inFile, dtype=np.float32, skip_header=1)    
    return (dataMat, dataSiz)
    
    
def loadDataFromIListFile(inFile):
    """Read list of image files into a data matrix.
    
    Reads a file with absolute image filenames, e.g.,
    
    /tmp/im0.png
    /tmp/im1.png
    ...
    
    into a data matrix with images as column vectors. 
    
    Parameters
    ----------
    inFile : string
        Input file name.
        
    Returns
    -------
    dataMat : numpy array, shape = (N, #images)
        Output data matrix, where N = (width x height)
        
    dataSiz : tuple of (height, width, #images)    
        The video dimensions.
    """
    import SimpleITK as sitk
    
    with open(inFile) as fid:
        fileNames = fid.readlines()
    
    dataMat = None
    for i, imFile in enumerate(fileNames):
        img = sitk.ReadImage(imFile.rstrip())        
        mat = sitk.GetArrayFromImage(img)
        
        if len(mat.shape) > 2:
            raise Exception("Only grayscale images are supported!")
        
        if dataMat is None:
            dataMat = np.zeros((np.prod(mat.shape),len(fileNames)))
        dataMat[:,i] = dat.reshape(-1)
    
    return (dataMat, (mat.shape[0], mat.shapep[1], len(fileNames)))    
        
        
        
        
        
