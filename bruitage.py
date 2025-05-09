import random
import numpy as np
import math
import nibabel as nib
import matplotlib.pyplot as plt
import sys
from scipy.stats import rice


def bruitage_salt_pepper(I):
    """
    Adds to the image a salt and pepper noise

    Parameters :
    I : numpy array containing the image we want to add the noise to

    Returns :
    I_new : numpy array of the noised image
    """
    n,m = I.shape
    I_new = I.copy()
    dynamic_range = np.max(I)
    for i in range(n):
        for j in range(m):
            I_new[i,j] = I[i,j] + random.randint(0,dynamic_range - I[i,j]) 
    return I_new  


def bruitage_racien(image,b = 0,loc=0,scale=1):
    """
    Adds to the image a racien noise

    Parameters :
    image : numpy array of the image
    b : shape parameter for b
    loc : for shifting the dencity function
    scale : for scaling the dencity function

    Returns :
    The noised image
    """ 
    noise = rice.rvs(b, loc=loc, scale=scale, size=image.shape)
    noisy_image = np.clip(image+noise, 0, 255)
    
    return noisy_image

"""
The data that we noise we know that are images
"""
def noise_all_data(fnoise,data):
    assert data.ndim == 3,"Wrong type of data"
    noised = []
    for im in data: 
        noised.append(fnoise(im))
    noised = np.array(noised)
    return noised   

