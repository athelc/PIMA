import dicom2nifti
import dicom2nifti.settings as settings
import os
import sys
import numpy as np

def make_dir(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def transform_data(destination_directory,source_directory):
    #Creattes the destinanion directory if it does not exist 
    make_dir(destination_directory)

    #form every directory we want to keep the slices that are in the horizontal direction SAG
    for name in os.listdir(source_directory):
        
        if '_SAG' in name :
            
            settings.disable_validate_slice_increment()
            settings.disable_validate_slicecount()
            settings.disable_validate_orientation()

            print(name)
            #we convert this data from .ima to .nii
            try:
                dicom2nifti.convert_directory(source_directory, destination_directory,reorient=True, compression=False)
            except:
                pass
    

nb_samples = 1
if __name__=="__main__":
    if len(sys.argv) > 2:
        destination_directory = sys.argv[1]
        source_directory =  sys.argv[2]
        for count, name in enumerate(sorted(os.listdir(source_directory))):
            if count > nb_samples : break
            
            dest = destination_directory+"/"+name
            src = source_directory+name+"/"+os.listdir(source_directory+"/"+name)[0]
            transform_data(dest,src)  
        
    else:
        print("Not enough arguments provided.")

