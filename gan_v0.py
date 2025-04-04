def train(data_loader,num_epochs,g_model,d_model,r_model,nb_feat):#
    
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=lr)
    optimizer_r = torch.optim.Adam(r_model.parameters(), lr=lr)
    
    g_model = g_model.to(device)
    d_model = d_model.to(device)
    r_model = r_model.to(device)
    
    #while loss small engouh
    
    for e in range(num_epochs):
        
        for i, (real_data,_) in enumerate(data_loader):
            
            #place it on the machine 
            real_data = real_data.to(device)
            #def separate to noised and denoised
            noised,denoised = separate_data(real_data)
            
            #labels
            fake_labels = torch.zeros(real_data.size(0), 1, device=device)
            real_labels = torch.ones(real_data.size(0), 1, device=device)
            
            
            ########################
            #Discriminator Training# 
            ########################
            
            optimizer_d.zero_grad()
            
            #optimize the discriminator by giving the real images
            
            real_output = d_model(real_data)
            
            #lossD_valid = fLoss(outputs, valid_labels)
            #lossD_real = (outputs).mean()
            #lossD_real.backward()
            
            #generate fake data (start by giving a random noise)
            z = torch.randn(batch_size, nb_feat, device=device)#TODO : fix the size depending of the dataloaders
            fake_data_noised = g_model(z)
            fake_data_denoised = r_model(noised_data)
            
            #stack fake_data_ = fake_data_noised + fake_data_denoised (format_data(noised, denoised))
            #We have 4 different types of data : 
            #(x,y):real data with labels 1 -> right/true
            
            #the labels for all of these will be zero -> wrong/false
            
            #maybe we don't need this (according to the formula given in the paper is not necessary)
            #(x_hat,y_hat):fake data
            #fake_fake = format_data(fake_data_noised,fake_data_denoised)
            
            #(x_hat,y) : fake real
            fake_real = format_data(fake_data_noised,denoised)
            fake_real_out = d_model(fake_real)
            
            #(x,y_hat) : real fake
            real_fake = format_data(noised,fake_data_denoised)
            real_fake_out = d_model(real_fake)
            #seperated_data =[fake_fake,fake_real,real_fake]
            #all_fake_data = torch.cat(seperated_data, dim=0)
            #fake_outputs = d_model(all_fake_data)
            
            #optimize the discriminator by giving the generated images

            d_loss_real = (tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real_output)).mean()
            d_loss_real_fake = (tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=real_fake_out)).mean()
            d_loss_fake_real = (tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake_real_out)).mean()

            #d_loss_real = FLoss(real_output,real_labels)
            #d_loss_real_fake = FLoss(real_fake_out,fake_labels)
            #d_loss_fake_real = FLoss(fake_real_out,fake_labels)

            
            d_loss = d_loss_real - (alpha)*d_loss_fake_real - (1-alpha)*d_loss_real_fake 
            
            d_loss.backward()
            optimizer_d.step()
            
            # Print losses
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
                  f'Discriminator Loss: {lossD_valid.item() + lossD_f.item():.4f}')
            
        ####################
        # Update Generator # 
        ####################
            
        optimizer_g.zero_grad()
        
        #we fix the denoiser and the discriminator
        
        fake_data_denoised = r_model(data)
        #fake_data = fake_data_noised + fake_data_denoised
        
        output = d_model(fake_data)
        
        loss_g = fLoss(outputs, fake_labels)
        loss_g.backward()
        optimizer_g.step()
            
        ###################
        # Update Denoizer # 
        ###################
            
        optimizer_r.zero_grad()
            
        #fake_data_noised = g_model(data)
        #fake_data = fake_data_noised + fake_data_denoised
        #output = d_model(fake_data)
        
        #loss_r = fLoss(outputs, fake_labels)
        #loss_r.backward()
        
        optimizer_r.step()
                