import torch

def format_data(noised, denoised):
    #assert noised.shape == denoised.shape, "Input tensors must have the same shape."
    if noised.shape[0] != denoised.shape[0]:
        dmin = min(noised.shape[0],denoised.shape[0])
        noised = noised[:dmin]
        denoised = denoised[:dmin]
        
    if noised.shape != denoised.shape:
        print(f"The shapes are : {noised.shape} and {denoised.shape}")
        raise ValueError(f"Input tensors must have the same shape. ")
    data = torch.stack((noised, denoised), dim=1)
    return data

def separate_data(data):
    assert data.shape[1] == 2, "Input tensor must have as second dim = 2 shape[1]"
    noised, denoised = torch.split(data, 1,dim=1)
    noised = noised.squeeze(1)
    denoised = denoised.squeeze(1)
    return noised, denoised 

