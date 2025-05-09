import numpy as np
import math
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import sys
from bruitage import *
from save_extr_dataloaders import *
import affichage
import torch



"""
Given a directory it creates 2 arrays 
the training : containing more or less 70% of the data 
the test : containing the rest 
For every file in that in the directory in gets the 30%of the images and adds them to the test array
"""
def extract_nii_data_in_patients(source_directory ,typeOfdata = '2_t2_tse_sag'):
    train = []
    test = []
    #we iterate in the main directory
    for name in os.listdir(source_directory):
        #iteration for each patient
        inner_directory = source_directory+"/"+name
        for filename in os.listdir(inner_directory):
            #extract only the type we want t1,t2 
            if typeOfdata in filename :
                print(filename)
                data_i = np.array((nib.load(inner_directory+"/"+filename)).dataobj) 
                #I want to get randomly 70%
                thirty = (data_i.shape[0]*30)//100

                indxes = random.sample(range(data_i.shape[0]), thirty)

                for i in range(data_i.shape[0]):
                    if i in indxes :
                        test.append(data_i[i])
                    else :
                        train.append(data_i[i])
                        
    return np.array(train),np.array(test)

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

if __name__=="__main__":

    batch_size = 8
    filename = "./25patients"
    
    if len(sys.argv) > 2:
        filename =  sys.argv[1] #"./25patients"
        batch_size = int(sys.argv[2])
        
    train_data,test_data = extract_nii_data_in_patients(filename)

    if filename == "./25patients" :
        test_data = np.concatenate((test_data,[train_data[-1]]), axis=0)
        train_data  = train_data[:-1]

    test_noised = noise_all_data(bruitage_racien,test_data)
    train_noised = noise_all_data(bruitage_racien,train_data)

    test_denoised_torch = torch.tensor(test_data, dtype=torch.float32) 
    test_noised_torch = torch.tensor(test_noised, dtype=torch.float32) 

    train_denoised_torch = torch.tensor(train_data, dtype=torch.float32) 
    train_noised_torch = torch.tensor(train_noised, dtype=torch.float32) 

    test_dataset = TensorDataset(test_denoised_torch, test_noised_torch)
    train_dataset = TensorDataset(train_denoised_torch, train_noised_torch)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    save_prompt = input("Do you want to save the results? (s/n): ").strip().lower()

    if save_prompt == 's':
        filename = input("Can you please give a filename ").strip()
            
        save_dataloader(test_dataloader,test_dataset, filename+'_test')
        save_dataloader(train_dataloader,train_dataset, filename+'_train')

        
    
 








