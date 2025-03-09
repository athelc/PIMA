import numpy as np
import math
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import bruitage 

def imshowmult(images=[],max_im = 5,save_image = False,ofilename = ""):
    
    max_images_per_row = max_im
    num_images = len(images)
    num_rows = math.ceil(num_images / max_images_per_row)
    
    fig, ax = plt.subplots(num_rows, max_images_per_row, figsize=(10, 3 * num_rows))
    
    ax = ax.flatten()

    for i in range(num_images):
        ax[i].imshow(np.rot90(images[i]), cmap='gray')
        ax[i].axis('off')  
    
    for i in range(num_images, len(ax)):
        ax[i].axis('off')
    
    if save_image :
        #ofilename = 'output_image.png' : example of ofilename
        plt.savefig(ofilename, dpi=300)
    plt.show()


def loadImg(filename):
    return np.array((nib.load(filename)).dataobj)

if __name__=="__main__":

    if len(sys.argv) > 1:
        img = loadImg(sys.argv[1])
        #imshowmult(img)
        img_salt_pepper = bruitage.bruitage_salt_pepper(img[0])
        img_racien = bruitage.bruitage_racien(img[0],1)
        imshowmult([img[0],img_salt_pepper,img[0],img_racien],2)


