import random
import numpy as np
import math
import nibabel as nib
import matplotlib.pyplot as plt
import sys
from scipy.stats import rice

def bruitage_salt_pepper(I):
    n,m = I.shape
    I_new = I.copy()
    dynamic_range = np.max(I)
    for i in range(n):
        for j in range(m):
            I_new[i,j] = I[i,j] + random.randint(0,dynamic_range - I[i,j]) 
    return I_new  


def bruitage_racien(image,b = 0,loc=0,scale=1):
    
    noise = rice.rvs(b, loc=loc, scale=scale, size=image.shape)
    noisy_image = np.clip(image+noise, 0, 255)
    
    return noisy_image
