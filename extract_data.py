import dicom2nifti
import dicom2nifti.settings as settings
import os
import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
import numpy as np

#Constant that represents the number of samples we want to extract 
#(the total that exists for our dataset is 574)
NB_SAMPLES = 25 #because we start to count at zero here we have 2 samples

def make_dir(directory_name):
    """
    Gets a directory_name and creates it.Returns the error in case of failure

    Parameters:
    directory_name(string) : the directory name we wish to have

    """
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
    """
    Finds the data from the files for a specific number of samples.
    Transforms them from .ima to .nii
    
    Parameters:
    destination_directory : the name of the directory we want out data to be extracted 
    source_directory : the name of the folder containing the data we want to extract
    """

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
                #dicom2nifti.dicom_series_to_nifti(source_directory, destination_directory, reorient_nifti=True)
                dicom2nifti.convert_directory(source_directory, destination_directory,reorient=True, compression=False)
            except:
                pass
    


if __name__=="__main__":

    if len(sys.argv) > 2:
        print("in the fun")
        
        source_directory =  sys.argv[1]
        destination_directory = sys.argv[2]
                                                                                                                        
        #for all the folders that we have in this directory we get and transform the first one
        for count, name in enumerate(sorted(os.listdir(source_directory))):
            if count > NB_SAMPLES  : break
            
            dest = destination_directory+"/"+name
            src = source_directory+name+"/"+os.listdir(source_directory+"/"+name)[0]
            transform_data(dest,src)  
        
    else:
        print("Not enough arguments provided.")