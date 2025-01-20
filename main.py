from utils import *
from mask import *

if __name__ == "__main__":
    filename = "./Data/InputData/training/images/21_training.tif"
    path = "./Data/InputData/training/images"

    images = readImg(path, lim=3)
    showImg(images[0][0], images[0][1])

    for ch in ["r","g","b","mean"]:
        grayim = toGray(images[0][0], channel=ch)
        showImg(grayim, images[0][1])