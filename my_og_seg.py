from utils import *
import skimage as sk
from skimage import io, draw
from skimage import morphology as mrph
from skimage import filters as flt
from skimage import transform as trf
import numpy as np

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
    seed[0:3,0:3] = 1 # Création d'une image de marqueur pour reconstruire l'extérieur

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

def OGSegmentVessels(im):
    """
    Segment the vessels in the image based on the mask image.
        -im: image in gray scale to segment
    RETURN
        -segIm = segmented image.
    """
    #Retrieve the mask of the image
    mask = imToMask(im)


    #Remove background noise
    im_med = flt.median(im)
    im_med = np.where(mask == 1, im_med, 0)
    #Reduce slow variation of the image's background luminance (bottom hat)
    ftp = np.zeros([9,9])
    ftp[:,4] = 1
    ftp[4,:] = 1
    im_bth = mrph.black_tophat(im_med, ftp)
    im_bth = im_bth/np.max(im_bth)
    im_bth = np.where(mask == 1, im_bth, 0)

    #Highlight vessels by a sequence of opening
        #Create footprints
    angles = np.linspace(0, 180, 30)
    ftp = np.zeros([21,21])
    ftp[:,10] = 1
    ftps = [trf.rotate(ftp, x) for x in angles]
        #Apply opening
    im_open = np.zeros(im.shape)
    for f in ftps:
        op = mrph.opening(im_bth, f)
        im_open += op

    im_open = im_open / np.max(im_open)
    im_open = np.where(mask == 1, im_open, 0)

    #Binarization of images
    im_bin = np.where(im_open >= 0.05, 1, 0)

    #Skeletonized
    im_skel = mrph.skeletonize(im_bin)
        #Remove artefacts
    im_skel_clean = mrph.area_opening(im_skel, area_threshold=150, connectivity=2)

    #Rebuild vessels
    im_reconstruct = mrph.reconstruction(im_skel_clean, im_bin, 'dilation')


    # #Reduce slow variation of image luminance (top-hat)
    # bthIm = mrph.black_tophat(im)
    # norm_bthIm = bthIm/np.max(bthIm)
    # # norm_bthIm = np.where(mask == 1, norm_bthIm, 0)

    # #Retrieve a background image
    # im_backmin = im - norm_bthIm
    # im_gauss = flt.gaussian(im_backmin, sigma=1)
    # im_median = flt.median(im_backmin)
    # im_bgauss = flt.difference_of_gaussians(im_backmin, low_sigma=1, high_sigma=5)

    #Visualization
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(im, cmap='gray')
    ax[0, 0].set_title('Original image')
    ax[0, 1].imshow(mask, cmap='gray')
    ax[0, 1].set_title('mask')
    ax[1, 0].imshow(im_bth, cmap='gray')
    ax[1, 0].set_title('bth')
    ax[1, 1].imshow(im_open, cmap='gray')
    ax[1, 1].set_title('open')
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(im, cmap='gray')
    ax[0, 0].set_title('Original image')
    ax[0, 1].imshow(im_bin, cmap='gray')
    ax[0, 1].set_title('binarized')
    ax[1, 0].imshow(im_skel_clean, cmap='gray')
    ax[1, 0].set_title('clean skeletonized')
    ax[1, 1].imshow(im_reconstruct, cmap='gray')
    ax[1, 1].set_title('reconstruction')
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.show()


    im_seg = sk.img_as_ubyte(im_reconstruct)
    return im_seg

