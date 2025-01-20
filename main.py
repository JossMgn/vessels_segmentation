from skimage import io
from matplotlib import pyplot as plt
import os
from random import randint

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
        -a list of red images (idx 0) and their name (idx 1)
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


if __name__ == "__main__":
    filename = "./Data/InputData/training/images/21_training.tif"
    path = "./Data/InputData/training/images"
    images = readImg(path, lim=3)
    for im in images:
        showImg(im[0], im[1])