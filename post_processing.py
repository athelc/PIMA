import os
import sys

if '__file__' in locals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()  

parent_dir = os.path.abspath(os.path.join(current_dir, 'GAN/'))

sys.path.append(current_dir)  
sys.path.append(parent_dir) 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_pre_processing import *
from generator_Denoiser import *
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mutual_info_score as mis
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

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
    axis[1].plot(np.arange(len(g_loss)), -g_loss, linewidth=1)
    axis[1].set_title("Generator Loss")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Loss")

    # Plotting the Denoiser Loss
    axis[2].plot(np.arange(len(r_loss)), -r_loss, linewidth=1)  
    axis[2].set_title("Denoiser Loss")
    axis[2].set_xlabel("Epochs")
    axis[2].set_ylabel("Loss")

    plt.tight_layout()
    plt.show()


def print_sample_of_result(denoised,noised,generator,denoiser):#

    #first_batch = next(iter(dataloader))
    #denoised, noised = first_batch
    batch_size = int(denoised.shape[0])
    nb = 5
    fig, axis = plt.subplots(2, 2)

    axis[0, 0].imshow(np.rot90(denoised[nb]), cmap='gray')
    axis[0, 0].set_title("Denoised Image")
    axis[0, 0].axis('off')  


    axis[1, 0].imshow(np.rot90(noised[nb]), cmap='gray')
    axis[1, 0].set_title("Noised Image") 
    axis[1, 0].axis('off') 


    t1 = torch.from_numpy(denoised[nb].numpy()).float()/255.0
    t1_expanded = t1.unsqueeze(0).expand(batch_size, -1, -1)

    t2 = torch.from_numpy(noised[nb].numpy()).float()/255.0
    t2_expanded = t2.unsqueeze(0).expand(batch_size, -1, -1)

    fake_im = (generator(t1_expanded).detach().numpy()).squeeze(1)
    den_im = (denoiser(t2_expanded).detach().numpy()).squeeze(1)

    
    axis[0,1].imshow(np.rot90(den_im[nb]), cmap='gray')
    axis[0, 1].set_title("Denoised form Denoiser Image")
    axis[0, 1].axis('off')

    axis[1,1].imshow(np.rot90(fake_im[nb]), cmap='gray')
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

    return np.mean(r_acc)*100,np.mean(g_acc)*100

"""Calculate the KL divergence between two normal distributions."""
def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    if np.any(sigma1 <= 0) or np.any(sigma2 <= 0):
        raise ValueError("Standard deviations must be positive.")
    
    kl_div = 0.5*(-np.log(sigma1**2 / sigma2**2) + (sigma1**2 + (mu1 - mu2)**2) / (sigma2**2) - 1)
    return kl_div

def AKLD_L_images(noised_batch,denoised_batch,generated_batch,L = 5):
    #take L random indices
    indx = random.sample(range(len(noised_batch)), L)
    akld = []
    for i in indx:
        real = denoised_batch[i]
        generated = generated_batch[i]
        noised = noised_batch[i]
        mu = real
        P = gaussian_filter((generated-real)**2, sigma=1)
        Q = gaussian_filter((noised-real)**2, sigma=1)
        k = kl_divergence_normal(mu,np.sqrt(np.diagonal(P)),mu,np.sqrt(np.diagonal(Q)))
        akld.append(k)
    return np.mean(akld)

def AKLD(dataloader,generator):
    n = len(dataloader)
    metric_gb = [] 

    for i,(batch_denoised, batch_noised) in enumerate(dataloader):
        print(f"calculating AKLD of batch {i+1}/{n} ")
        generated = normalize((generator(batch_denoised).detach().numpy()).squeeze(1))
        denoised = normalize(batch_denoised.numpy())
        noised = normalize(batch_noised.numpy())

        metric_gb.append(AKLD_L_images(noised,denoised,generated))
    
    return metric_gb

def ssim_fro_batch(original_batch,modified_batch):
    ssim_tot = []
    for original,modified in zip(original_batch,modified_batch):
        ssim_tot.append(ssim(original ,modified,data_range=modified.max() - modified.min()))

    return ssim_tot

def psnr_fro_batch(original_batch,modified_batch):
    psnr_tot = []
    for original,modified in zip(original_batch,modified_batch):
        psnr_tot.append(psnr(original ,modified))

    return psnr_tot#np.mean(psnr_tot)

def mse_for_batch(original_batch,modified_batch):
    mse_tot = []
    for original,modified in zip(original_batch,modified_batch):
        mse_tot.append(mse(original ,modified))

    return mse_tot#np.mean(mse_tot)


def im_for_batch(original_batch,modified_batch):
    im = []
    for original,modified in zip(original_batch,modified_batch):
        assert original.shape == modified.shape,"The images have to be the same size"
        flat_org = (original.flatten() * 255).astype(np.uint8)
        flat_mod = (modified.flatten()* 255).astype(np.uint8)
 
        im.append(mis(flat_org,flat_mod))

    return im#np.mean(im)

def blur_for_batch(original_batch,modified_batch):
    im = []
    for img in modified_batch:
        im.append(cv2.Laplacian(img, cv2.CV_32F).var())
    return np.array(im)

#general function to calculate the metrics on the dataloader
def metric(dataloader,generator,denoiser,batch_fun):
    n = len(dataloader)
    metric_gb = []
    metric_rb = []  

    for i,(batch_denoised, batch_noised) in enumerate(dataloader):
        print(f"calculating metric of batch {i+1}/{n} ")
        generated = normalize((generator(batch_denoised).detach().numpy()).squeeze(1))
        cleaned = normalize((denoiser(batch_noised).detach().numpy()).squeeze(1))
        
        denoised = normalize(batch_denoised.numpy())
        noised = normalize(batch_noised.numpy())
        r = batch_fun(denoised,cleaned)
        g = batch_fun(noised,generated)
        metric_rb.append(r)
        metric_gb.append(g)
    #Question : should I take also the avg of the std of each bach
    return np.mean(metric_rb),np.std(metric_rb),np.mean(metric_gb),np.std(metric_gb)



if __name__=="__main__":

    test_dataloader,test_batch_size = load_dataloader('./datald/25pat_8_test',False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = GRModel(240,1)
    denoiser = GRModel(240,1)
    
    generator = generator.to(device)
    denoiser = denoiser.to(device)

    generator1 = GRModel(240,1)
    denoiser1 = GRModel(240,1)
    
    generator1 = generator.to(device)
    denoiser1 = denoiser.to(device)

    generator.load_state_dict(torch.load('./models/230_7_batch_10_epochs/g_model_230.pth', map_location=torch.device('cpu')))
    denoiser.load_state_dict(torch.load('./models/230_7_batch_10_epochs/r_model_230.pth', map_location=torch.device('cpu')))

    generator1.load_state_dict(torch.load('./models/t100_n5/g_model.pth', map_location=torch.device('cpu')))
    denoiser1.load_state_dict(torch.load('./models/t100_n5/r_model.pth', map_location=torch.device('cpu')))

    first_batch = next(iter(test_dataloader))
    denoised, noised = first_batch

    generated = normalize((generator(denoised).detach().numpy()).squeeze(1))
    cleaned = normalize((denoiser(noised).detach().numpy()).squeeze(1))
        
    denoised = normalize(denoised.numpy())
    noised = normalize(noised.numpy())
    #print_sample_of_result(denoised,noised,generator,denoiser)
    #print_sample_of_result(denoised,noised,generator1,denoiser1)
    #print("The accuracy : (denoiser,generator)",accuracy(test_dataloader,generator1,denoiser1))
    #print(metric(test_dataloader,generator,denoiser,mse_for_batch))
    #print(metric(test_dataloader,generator,denoiser,ssim_fro_batch))
    #print(metric(test_dataloader,generator,denoiser, blur_for_batch))
    #print(AKLD(test_dataloader,generator))
    #training_results('./models/230_7_batch_10_epochs')
    #training_results('./models/t100_n5')
    print("This is the blur of the original images")
    print(blur_for_batch(noised,denoised))
    print("This is the blur of the original blured images")
    print(blur_for_batch(noised,noised))
    print("This is the blur of the denoised images")
    print(blur_for_batch(noised,cleaned))