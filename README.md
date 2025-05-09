# PIMA

# How to Use

## Data Extraction
The first step is to download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/k57fr854j2/2). Once you have downloaded the dataset, we recommend placing it in the same folder. Then, create a folder where you would like to store your extracted data. After that, execute the following command:

```bash
#python3 extract_data.py path_to_dataset path_to_destination_folder
python extract_data.py ./MRI_Data/01_MRI_Data/ ./test
```
## Visualization
visualize the data, you need to provide a .nii file as input. Use the following command:
```bash
python affichage.py ./test/0001/2_t2_tse_sag_384.nii
```
Make sure to replace the path with the actual path to the .nii file you wish to visualize.

## How to pre-prossess the data
First you have to transform the data from nii to numpy arrays as menttioned before. Then you can have to create the dataloaders. Tow dataloaders will be created with the execution of th `data_pre_processing.py`. One of them will contain the 70% of the normal and noised images and the other one will contain the rest.
When you execute this file you have to give the filename that contains the transforned data (the ones that you transformed earlier) and a batch size. If you don't give anything then it will create the dataloaders of the file *./25patiens* and a batch size *8*.
after the creation it will ask if you want to save them if you wish you should give the filename you want to save them in.

## How to train
Once you have saved your dataloaders you can now train your data.