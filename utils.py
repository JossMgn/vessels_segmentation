from skimage import io
from matplotlib import pyplot as plt
import os
from random import randint
import numpy as np

def showImg(image, imname=None):
    '''
        Allow to display image with a legend or not.
        -image: the image that you want to display
        -imname: the name of the image in the db
    '''
    io.imshow(image)
    if imname is not None:
        plt.title(imname)
    plt.show()

def readImg(path, lim=-1, rand=True):
    '''
        Read image(s) and display it at the end. You could chose to display all the db or a limited number of images.
        -path: the path of the directory with images
        -lim: the number max of images to display (-1 for all images, else chose a number)
        -rand: allow to choose random images from the db or not (if not, the first images will be display)
        RETURN
        -a list of red images as ndarray(idx 0) and their name (idx 1)
    '''
    filenames = os.listdir(path)

    if lim > 0 and lim < len(filenames):
        if rand == True:
            indexes = []
            for i in range(lim):
                new_idx = randint(0, len(filenames)-1)
                while new_idx in indexes:
                    new_idx = randint(0, len(filenames)-1)
                indexes.append(new_idx)
            filenames = [filenames[i] for i in indexes]
        else:
            filenames = filenames[0:lim]

    filepaths = [os.path.join(path, name) for name in filenames]
    images = []
    for fp, fn in zip(filepaths, filenames):
        images.append([io.imread(fp), fn])

    return images

def compare_img(gt_img, seg_img):
    """
        Return the IoU between the segmented image and the ground truth.
    """
    #iou 
    intersection = np.sum(np.where(gt_img==255, seg_img, 0) == 255) #give also the true positive
    union = np.sum(gt_img == 255) + np.sum(seg_img == 255) - intersection
    iou = intersection / union

    #FP
    fp = np.sum(seg_img == 255) - intersection

    #TN
    tn = np.sum(np.where(gt_img==0, seg_img, 255)==0)

    #FN
    fn = np.sum(np.where(gt_img==1, seg_img, 255)==0)

    print(iou, intersection, fp, tn, fn)