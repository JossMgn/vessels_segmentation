from utils import *
from mask import *
import time

if __name__ == "__main__":
    filename = "./Data/InputData/training/images/21_training.tif"
    path_img = "./Data/InputData/training/images"
    path_gt = "./Data/InputData/training/1st_manual"

    start = time.time()
    images = readImg(path_img, lim=3)
    end = time.time()
    gt = readImg(path_gt, lim=3)
    print(f"time for reading images: {round(end-start, 3)}sec.")
    # showImg(images[0][0], images[0][1])

    # for ch in ["r","g","b","mean"]:
    #     grayim = toGray(images[0][0], channel=ch)
    #     showImg(grayim, images[0][1])

    # start = time.time()
    # grayim = toGray(images[0][0], channel="g")
    # mask = imToMask(grayim)
    # end = time.time()
    # print(f"time to create the mask from the colored image: {round(end-start, 3)}sec.")

    # start = time.time()
    grayim = toGray(images[0][0], channel="g")
    # plt.imshow(gt[0][0][0])
    # plt.show()
    segmentVessels(grayim)
    compare_img(gt[0][0][0], gt[0][0][0])
    # end = time.time()

    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(images[0][0])
    # axs[0, 0].set_title(images[0][1])
    # axs[0, 1].imshow(grayim, cmap='gray')
    # axs[0, 1].set_title('Image in grayscale ch=g')
    # axs[1, 0].imshow(mask)
    # axs[1, 0].set_title('mask')
    # for a in axs.ravel():
    #     a.axis('off')
    # plt.tight_layout()
    # plt.show()

