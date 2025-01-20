from utils import *
from skimage import io
import numpy as np

def toGray(im, channel):
    '''
    Convert a colored image into a gray scale image.
    -im: the image to convert
    -channel: the method to convert:    "r" -> take only the red channel
                                        "g" -> take only the green channel
                                        "b" -> take only the blue channel
                                        "mean" -> do the mean of the three channels
    RETURN
    -grayim: the one channel image
    '''
    channel_colors = ["r","g","b"]
    if channel in channel_colors:
        grayim = im[:,:,channel_colors.index(channel)]
    elif channel == "mean":
        grayim = np.mean(im, axis=2)
    
    return grayim