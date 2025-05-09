import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,nb_feat):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            #Convolutional layer
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            
            #Convolutional layer
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            #Convolutional layer
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256,1, kernel_size=4, stride=2, padding=1),
            
            nn.Flatten(),
            nn.Linear(24 * 24, 1),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        return self.main(x)