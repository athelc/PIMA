import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(current_dir)  
sys.path.append(parent_dir) 

import numpy as np
from save_extr_dataloaders import *
from necessary_functions import * 
from discriminator import *
from generator_Denoiser import *

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
NB_FEATURES = 384

def train(data_loader,num_epochs,g_model,d_model,r_model,nb_feat,alpha,n_critic,loss_fn):
    
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=lr)
    optimizer_r = torch.optim.Adam(r_model.parameters(), lr=lr)
    
    g_model = g_model.to(device)
    d_model = d_model.to(device)
    r_model = r_model.to(device)
    
    #while loss small engouh
    d_loss_hist = []
    g_loss_hist = []
    r_loss_hist = []
    for e in range(num_epochs):
        print(f"epoch {e} done")
        for n in range(n_critic):
            avg_loss = []
            print(f"critic {n} done")
            
            for i, (denoised,noised) in enumerate(data_loader):

                #place it on the machine 
                noised = noised.to(device)
                denoised = denoised.to(device)
                #def separate to noised and denoised

                #labels
                fake_labels = torch.zeros(noised.size(0), 1, device=device)
                real_labels = torch.ones(noised.size(0), 1, device=device)


                ########################
                #Discriminator Training# 
                ########################

                optimizer_d.zero_grad()

                #optimize the discriminator by giving the real images
                #print("this done 1")
                real_data = format_data(noised,denoised)
                real_output = d_model(real_data)

                #generate fake data (start by giving a random noise)
                z = torch.randn(batch_size, nb_feat,nb_feat, device=device)#TODO : fix the size depending of the dataloaders
                fake_data_noised = (g_model(z)).squeeze(1) 
                fake_data_denoised = (r_model(noised)).squeeze(1)
                #print("this done 2")
                #(x_hat,y) : fake real
                fake_real = format_data(fake_data_noised,denoised)
                fake_real_out = d_model(fake_real)
                #print("this done 3")
                #(x,y_hat) : real fake
                real_fake = format_data(noised,fake_data_denoised)
                real_fake_out = d_model(real_fake)
                #ipdb.set_trace()
                #optimize the discriminator by giving the generated images

                d_loss_real = loss_fn(real_output,real_labels)
                d_loss_real_fake = loss_fn(real_fake_out,fake_labels)
                d_loss_fake_real = loss_fn(fake_real_out,fake_labels)

                d_loss = d_loss_real - (alpha)*d_loss_fake_real - (1-alpha)*d_loss_real_fake 
                #keep the value for statistics later
                avg_loss.append(d_loss.item())
                
                d_loss.backward()
                optimizer_d.step()
                #ipdb.set_trace()
                # Print losses
                if i % 100 == 0:
                    print(f'Epoch [{e+1}/{num_epochs}], Step [{n}/{n_critic}], '
                      f'Discriminator Loss: {d_loss.item():.4f}')
        
        if avg_loss :
            d_loss_hist.append(sum(avg_loss)/len(avg_loss))
        print("We go here")
        
        r_avg_loss = []
        g_avg_loss = []
        for i, (denoised,noised) in enumerate(data_loader):

            #place it on the machine 
            noised = noised.to(device)
            denoised = denoised.to(device)

            #labels
            fake_labels = torch.zeros(noised.size(0), 1, device=device)
            real_labels = torch.ones(noised.size(0), 1, device=device)


            ####################
            # Update Generator # 
            ####################

            optimizer_g.zero_grad()


            fake_data_denoised = (r_model(noised)).squeeze(1)
            
            #fake_data = fake_data_noised + fake_data_denoised
            real_fake = format_data(noised,fake_data_denoised)
            real_fake_out = d_model(real_fake)


            g_loss =  alpha*loss_fn(real_fake_out,fake_labels)
            
            g_avg_loss.append(g_loss.item())
            
            g_loss.backward()
            optimizer_g.step()

            ###################
            # Update Denoizer # 
            ###################

            optimizer_r.zero_grad()

            fake_data_noised = (g_model(denoised)).squeeze(1)
            fake_real = format_data(fake_data_noised,denoised)
            fake_real_out = d_model(fake_real)

            r_loss =(1-alpha)*loss_fn(fake_real_out,fake_labels)
            r_avg_loss.append(r_loss.item())
            
            r_loss.backward()
            optimizer_r.step()
            
            if i % 100 == 0:
                print(f'Epoch [{e+1}/{num_epochs}], '
                      f'R : Denoiser Loss: {r_loss.item():.4f}')
        if g_avg_loss :
            g_loss_hist.append(sum(g_avg_loss)/len(g_avg_loss))

        if r_avg_loss :
            r_loss_hist.append(sum(r_avg_loss)/len(r_avg_loss))
    return d_loss_hist,r_loss_hist,g_loss_hist



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

    torch.save(discriminator.state_dict(), out_filename+'/d_model.pth')
    torch.save(generator.state_dict(), out_filename+'/g_model.pth')
    torch.save(denoiser.state_dict(), out_filename+'/r_model.pth')
    np.savez(out_filename+'/graph_results.npz', array1=d_loss_hist, array2=r_loss_hist, array3=g_loss_hist)

    