import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nb_feat):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1), # Réduction encore de filtres 16 au lieu de 32 et canaux d'entrée = 2
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 16 32 au lieu de 32 64
            nn.BatchNorm2d(32), # 32 et non 64
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5), # Augmentation de dropout 0.3, 0.5 

            nn.Conv2d(32, 1, kernel_size=4, stride=2, padding=1), # 32 au lieu de 64
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(2304, 1),  # Change the input dimension here
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)