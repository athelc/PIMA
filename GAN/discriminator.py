import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nb_feat):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(2304, 1),  # Change the input dimension here
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)