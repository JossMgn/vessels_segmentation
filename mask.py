from utils import *
from skimage import io, draw
from skimage import morphology as mrph
from skimage import filters as flt
import numpy as np
from example import *

def toGray(im, channel):
    '''
    Convert a colored image into a gray scale image.
    -im: the image to convert
    -channel: the method to convert:    \n\t"r" -> take only the red channel
                                        \n\t"g" -> take only the green channel
                                        \n\t"b" -> take only the blue channel
                                        \n\t"mean" -> do the mean of the three channels
    RETURN
    -grayim: the one channel image
    '''
    channel_colors = ["r","g","b"]
    if channel in channel_colors:
        grayim = im[:,:,channel_colors.index(channel)]
    elif channel == "mean":
        grayim = np.mean(im, axis=2)

    grayim = (grayim/255).astype(np.float32)
    return grayim

def imToMask(im):
    '''
    -im: gray image use to create the mask.\n
    RETURN
    -mask: the mask of the im in output.
    '''
    edges = flt.sobel(im)

    low = 0.01
    high = 0.1

    hyst = flt.apply_hysteresis_threshold(edges, low, high) #détection des bords par hysteresis

    x, y = np.shape(im)
    seed = np.zeros((x, y), dtype=np.float32)
    seed[0:50,0:50] = 0.5 # Création d'une image de marqueur pour reconstruire l'extérieur

    mask = mrph.reconstruction(seed, 1-hyst, 'dilation') #Utilisation de la dilatation géodésique pour reconstruire les bords extérieurs
    
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0, 0].imshow(im, cmap='gray')
    # ax[0, 0].set_title('Original image')
    # ax[0, 1].imshow(edges, cmap='magma')
    # ax[0, 1].set_title('Sobel edges')
    # ax[1, 0].imshow(hyst, cmap='magma')
    # ax[1, 0].set_title('hysteresis')
    # ax[1, 1].imshow(mask, cmap='magma')
    # ax[1, 1].set_title('mask')
    # for a in ax.ravel():
    #     a.axis('off')
    # plt.tight_layout()
    # plt.show()

    return 1-mask


