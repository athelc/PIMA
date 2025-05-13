import torch
from torch.utils.data import TensorDataset , DataLoader
torch.serialization.add_safe_globals([TensorDataset])


def save_dataloader(dataloader,dataset,filename):
    torch.save(dataset, filename+'.pth')  # Save the dataset
    dataloader_params = {
        'batch_size': dataloader.batch_size,
        'shuffle': True,#dataloader.shuffle,
    }
    torch.save(dataloader_params, filename+'_params.pth')  # Save parameters

def load_dataloader(filename,shuffle = True):
    # Load the dataset
    loaded_dataset = torch.load(filename+'.pth', weights_only=True)  # Load the dataset

    # Load the DataLoader parameters
    loaded_dataloader_params = torch.load(filename+'_params.pth')  # Load parameters
    b_size = loaded_dataloader_params['batch_size']
    # Recreate the DataLoader
    loaded_dataloader = DataLoader(
        loaded_dataset,
        batch_size=b_size,
        shuffle=shuffle#loaded_dataloader_params['shuffle']
    )
    
    return loaded_dataloader,b_size