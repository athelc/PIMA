import os
import sys

if '__file__' in locals():
    current_dir = os.path.dirname(__file__)
else:
    current_dir = os.getcwd()  

parent_dir = os.path.abspath(os.path.join(current_dir, 'GAN/'))

sys.path.append(current_dir)  
sys.path.append(parent_dir) 

#import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_pre_processing import *
from generator_Denoiser import *
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mutual_info_score as mis
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim


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


    print_sample_of_result(denoised,noised,generator,denoiser)
    #print_sample_of_result(denoised,noised,generator1,denoiser1)
    #print("The accuracy : (denoiser,generator)",accuracy(test_dataloader,generator1,denoiser1))
    #print(metric(test_dataloader,generator,denoiser,mse_for_batch))
    #print(metric(test_dataloader,generator,denoiser,im_for_batch))
    #print(metric(test_dataloader,generator,denoiser,ssim_fro_batch))
    #print(metric(test_dataloader,generator,denoiser, blur_for_batch))
    #print(AKLD(test_dataloader,generator))
    #training_results('./models/230_7_batch_10_epochs')
    #training_results('./models/t100_n5')
    """generated = normalize((generator(denoised).detach().numpy()).squeeze(1))
    cleaned = normalize((denoiser(noised).detach().numpy()).squeeze(1))
        
    denoised = normalize(denoised.numpy())
    noised = normalize(noised.numpy())"""


    #print(AKLD_L_images(noised,denoised,noised,L = 5))
    """print("This is the blur of the original images")
    print(blur_for_batch(noised,denoised))
    print("This is the blur of the original blured images")
    print(blur_for_batch(noised,noised))
    print("This is the blur of the denoised images")
    print(blur_for_batch(noised,cleaned))"""