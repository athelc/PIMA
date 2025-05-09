import os
import sys

if '__file__' in locals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()  

parent_dir = os.path.abspath(os.path.join(current_dir, 'GAN/'))

sys.path.append(current_dir)  
sys.path.append(parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from data_pre_processing import *
from generator_Denoiser import *
def training_results(filename):
    #gets the information of the 
    data = np.load(filename+'/graph_results.npz')

    # Extract the arrays
    d_loss = data['array1']
    r_loss = data['array2']
    g_loss = data['array3']

    #assert len(d_loss) == len(r_loss) and len(r_loss) == len(d_loss) , "Lengths of losses are not equal"

    figure, axis = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting the Discriminator Loss
    axis[0].plot(np.arange(len(d_loss)), d_loss, linewidth=1) 
    axis[0].set_title("Discriminator Loss")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Loss")

    # Plotting the Generator Loss
    axis[1].plot(np.arange(len(g_loss)), g_loss, linewidth=1)
    axis[1].set_title("Generator Loss")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Loss")

    # Plotting the Denoiser Loss
    axis[2].plot(np.arange(len(r_loss)), r_loss, linewidth=1)  
    axis[2].set_title("Denoiser Loss")
    axis[2].set_xlabel("Epochs")
    axis[2].set_ylabel("Loss")

    plt.tight_layout()
    plt.show()


def print_sample_of_result(dataloader,generator,denoiser):

    first_batch = next(iter(dataloader))
    denoised, noised = first_batch
    batch_size = int(denoised.shape[0])

    fig, axis = plt.subplots(2, 2)

    axis[0, 0].imshow(np.rot90(denoised[0]), cmap='gray')
    axis[0, 0].set_title("Denoised Image")
    axis[0, 0].axis('off')  


    axis[1, 0].imshow(np.rot90(noised[0]), cmap='gray')
    axis[1, 0].set_title("Noised Image") 
    axis[1, 0].axis('off') 


    t1 = torch.from_numpy(denoised[0].numpy()).float()/255.0
    t1_expanded = t1.unsqueeze(0).expand(batch_size, -1, -1)

    t2 = torch.from_numpy(noised[0].numpy()).float()/255.0
    t2_expanded = t2.unsqueeze(0).expand(batch_size, -1, -1)

    fake_im = (generator(t1_expanded).detach().numpy()).squeeze(1)
    den_im = (denoiser(t2_expanded).detach().numpy()).squeeze(1)

    
    axis[0,1].imshow(np.rot90(den_im[0]), cmap='gray')
    axis[0, 1].set_title("Denoised form Denoiser Image")
    axis[0, 1].axis('off')

    axis[1,1].imshow(np.rot90(fake_im[0]), cmap='gray')
    axis[1, 1].set_title("Generated Image")
    axis[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def normalize(X):
    maxi = np.max(X)
    mini = np.min(X)
    X = (X - mini)/(maxi- mini)
    return X

def accuracy(dataloader,generator,denoiser):
    r_acc = []
    g_acc = []
    n = len(dataloader)
    for i,(batch_denoised, batch_noised) in enumerate(dataloader):
        print(f"calculating accuracy of batch {i+1}/{n} ")
        fake_im = normalize((generator(batch_denoised).detach().numpy()).squeeze(1))
        den_im = normalize((denoiser(batch_noised).detach().numpy()).squeeze(1))
        
        norm_den = normalize(batch_denoised.numpy())
        norm_noi = normalize(batch_noised.numpy())
        
        r_acc.append(np.mean(np.abs(norm_den - den_im)))
        g_acc.append(np.mean(np.abs(norm_noi - fake_im)))

    return np.mean(r_acc),np.mean(g_acc)



if __name__=="__main__":

    test_dataloader,test_batch_size = load_dataloader('./datald/25pat_8_test')

    generator = GRModel(240,1)
    denoiser = GRModel(240,1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    denoiser = denoiser.to(device)

    generator.load_state_dict(torch.load('./models/230_7_batch_10_epochs/g_model_230.pth', map_location=torch.device('cpu')))
    denoiser.load_state_dict(torch.load('./models/230_7_batch_10_epochs/r_model_230.pth', map_location=torch.device('cpu')))

    #generator.load_state_dict(torch.load('./models/g_model.pth', map_location=torch.device('cpu')))
    #denoiser.load_state_dict(torch.load('./models/r_model.pth', map_location=torch.device('cpu')))


    print_sample_of_result(test_dataloader,generator,denoiser)
    #print("The accuracy : (denoiser,generator)",accuracy(test_dataloader,generator,denoiser))
    training_results('./models/230_7_batch_10_epochs')
    #training_results('./models/')