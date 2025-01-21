from utils import *
from mask import *
import time

if __name__ == "__main__":
    filename = "./Data/InputData/training/images/21_training.tif"
    path = "./Data/InputData/training/images"

    start = time.time()
    images = readImg(path, lim=3)
    end = time.time()
    print(f"time for reading images: {round(end-start, 3)}sec.")
    # showImg(images[0][0], images[0][1])

    # for ch in ["r","g","b","mean"]:
    #     grayim = toGray(images[0][0], channel=ch)
    #     showImg(grayim, images[0][1])
    start = time.time()
    grayim = toGray(images[0][0], channel="g")
    mask = imToMask(grayim)
    end = time.time()
    print(f"time to create the mask from the colored image: {round(end-start, 3)}sec.")

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(images[0][0])
    axs[0, 0].set_title(images[0][1])
    axs[0, 1].imshow(grayim, cmap='gray')
    axs[0, 1].set_title('Image in grayscale ch=g')
    axs[1, 0].imshow(mask)
    axs[1, 0].set_title('mask')
    for a in axs.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.show()

