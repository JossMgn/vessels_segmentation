from utils import *
from my_og_seg import *
import time

if __name__ == "__main__":
    filename = "./Data/InputData/training/images/21_training.tif"
    path_img = "./Data/InputData/training/images"
    path_gt = "./Data/InputData/training/1st_manual"

    start = time.time()
    images = readImg(path_img, lim=3)
    end = time.time()
    # gt = readImg(path_gt, lim=3)
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
    im_seg = OGSegmentVessels(grayim)
    # end = time.time()
    gt = readGt(images[0][1], path_gt)
    compare_img(gt[0][0], im_seg)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(images[0][0])
    axs[0, 0].set_title(images[0][1])
    axs[0, 1].imshow(im_seg, cmap='gray')
    axs[0, 1].set_title('Image segmented')
    axs[1, 0].imshow(gt[0][0])
    axs[1, 0].set_title(gt[1])
    for a in axs.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.show()

