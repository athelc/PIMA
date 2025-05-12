import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(current_dir)  
sys.path.append(parent_dir) 

import numpy as np
from save_extr_dataloaders import *
from necessary_functions import * 
from GAN.discriminator import *
from generator_Denoiser import *

import matplotlib.pyplot as plt

def plot_losses(d_loss_hist, r_loss_hist, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_hist, label='Discriminator Loss')
    plt.plot(r_loss_hist, label='Denoiser Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder la figure
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()


device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
lr_d = 0.00005  
NB_FEATURES = 384

def train(data_loader, num_epochs, g_model, d_model, r_model, nb_feat, alpha, n_critic, loss_fn, lr=0.0002):
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=lr_d)
    optimizer_r = torch.optim.Adam(r_model.parameters(), lr=lr)

    g_model = g_model.to(device)
    d_model = d_model.to(device)
    r_model = r_model.to(device)

    d_loss_hist = []
    g_loss_hist = []
    r_loss_hist = []

    for e in range(num_epochs):
        print(f"epoch {e} done")
        for n in range(n_critic):
            avg_loss = []
            print(f"critic {n} done")

            for i, (denoised, noised) in enumerate(data_loader):
                noised = noised.to(device)
                denoised = denoised.to(device)

                fake_labels = torch.zeros(noised.size(0), 1, device=device)*0.9# faire un label smoothing *0.9 
                real_labels = torch.ones(noised.size(0), 1, device=device)*0.1# faire un label smoothing *0.1

                ########################
                # Discriminator Training #
                ########################
                optimizer_d.zero_grad()

                real_data = format_data(noised, denoised)
                real_output = d_model(real_data)

                z = torch.randn(batch_size, nb_feat, nb_feat, device=device)
                fake_data_noised = (g_model(z)).squeeze(1)
                fake_data_denoised = (r_model(noised)).squeeze(1)

                fake_real = format_data(fake_data_noised, denoised)
                fake_real_out = d_model(fake_real)

                real_fake = format_data(noised, fake_data_denoised)
                real_fake_out = d_model(real_fake)

                d_loss_real = loss_fn(real_output, real_labels)
                d_loss_real_fake = loss_fn(real_fake_out, fake_labels)
                d_loss_fake_real = loss_fn(fake_real_out, fake_labels)

                d_loss = d_loss_real - (alpha)*d_loss_fake_real - (1-alpha)*d_loss_real_fake
                avg_loss.append(d_loss.item())

                d_loss.backward()
                optimizer_d.step()

                if i % 100 == 0:
                    print(f'Epoch [{e+1}/{num_epochs}], Step [{n}/{n_critic}], '
                          f'Discriminator Loss: {d_loss.item():.4f}')

        if avg_loss:
            d_loss_hist.append(sum(avg_loss)/len(avg_loss))
        print("We go here")

        r_avg_loss = []
        g_avg_loss = []
        for i, (denoised, noised) in enumerate(data_loader):
            noised = noised.to(device)
            denoised = denoised.to(device)

            fake_labels = torch.zeros(noised.size(0), 1, device=device)
            real_labels = torch.ones(noised.size(0), 1, device=device)

            ####################
            # Update Generator #
            ####################
            optimizer_g.zero_grad()

            fake_data_denoised = (r_model(noised)).squeeze(1)
            real_fake = format_data(noised, fake_data_denoised)
            real_fake_out = d_model(real_fake)

            g_loss = alpha*loss_fn(real_fake_out, fake_labels)
            g_avg_loss.append(g_loss.item())

            g_loss.backward()
            optimizer_g.step()

            ###################
            # Update Denoizer #
            ###################
            optimizer_r.zero_grad()

            fake_data_noised = (g_model(denoised)).squeeze(1)
            fake_real = format_data(fake_data_noised, denoised)
            fake_real_out = d_model(fake_real)

            r_loss = (1-alpha)*loss_fn(fake_real_out, fake_labels)
            r_avg_loss.append(r_loss.item())

            r_loss.backward()
            optimizer_r.step()

            if i % 100 == 0:
                print(f'Epoch [{e+1}/{num_epochs}], '
                      f'R : Denoiser Loss: {r_loss.item():.4f}')

        if g_avg_loss:
            g_loss_hist.append(sum(g_avg_loss)/len(g_avg_loss))

        if r_avg_loss:
            r_loss_hist.append(sum(r_avg_loss)/len(r_avg_loss))

        # Sauvegarder les modèles à intervalles réguliers
        if e % 10 == 0:
            torch.save(g_model.state_dict(), f'g_model_epoch_{e}.pth')
            torch.save(d_model.state_dict(), f'd_model_epoch_{e}.pth')
            torch.save(r_model.state_dict(), f'r_model_epoch_{e}.pth')

        # Early stopping
        if len(d_loss_hist) > 10 and d_loss_hist[-1] > d_loss_hist[-10]:
            print("Early stopping")
            break

    return d_loss_hist, r_loss_hist, g_loss_hist

if __name__=="__main__":
    generator = GRModel(240,1)
    denoiser = GRModel(240,1)
    discriminator = Discriminator(NB_FEATURES)

    generator = generator.to(device)
    denoiser = denoiser.to(device)
    discriminator = discriminator.to(device)

    filename = input("please give the filename of the data : ").strip() #'train_dataset'
    dataloader,batch_size = load_dataloader(filename)

    out_filename = input("please give a destination filename : ").strip()

    nb_epochs = int(input("Please give a the number of epochs you want : "))
    if nb_epochs <= 0 or nb_epochs > 2000 :
        raise ValueError("Wrong nb of epochs")

    alpha = float(input("Enter the alpha wished between 0 and 1 ex. 0.5 : "))
    if alpha <= 0 or alpha > 1 :
        raise ValueError("It has to be between 0 and 1")

    n_critic = int(input("Please give a the number of n_critics you want : "))
    if n_critic <= 0 or n_critic > 2000 :
        raise ValueError("Wrong nb of critic")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    d_loss_hist,r_loss_hist,g_loss_hist = train(dataloader,nb_epochs,generator,discriminator,denoiser,NB_FEATURES,alpha,n_critic,loss_fn)
    # Appeler la fonction de traçage après l'entraînement
    plot_losses(d_loss_hist, r_loss_hist, out_filename)

    torch.save(discriminator.state_dict(), out_filename+'/d_model.pth')
    torch.save(generator.state_dict(), out_filename+'/g_model.pth')
    torch.save(denoiser.state_dict(), out_filename+'/r_model.pth')
    np.savez(out_filename+'/graph_results.npz', array1=d_loss_hist, array2=r_loss_hist, array3=g_loss_hist)

    