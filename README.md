# PIMA

# How to Use

## Data Extraction
The first step is to download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/k57fr854j2/2). Once you have downloaded the dataset, we recommend placing it in the same folder. Then, create a folder where you would like to store your extracted data. After that, execute the following command:

```bash
#python3 extract_data.py path_to_dataset path_to_destination_folder
python extract_data.py ./MRI_Data/01_MRI_Data ./test
```
## Visualization
visualize the data, you need to provide a .nii file as input. Use the following command:
```bash
python affichage.py ./test/0001/2_t2_tse_sag_384.nii
```
Make sure to replace the path with the actual path to the .nii file you wish to visualize.