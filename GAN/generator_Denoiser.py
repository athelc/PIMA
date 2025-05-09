import torch.nn as nn

class GRModel(nn.Module):
    def __init__(self, input_dim,out_dim):
        super(GRModel, self).__init__()
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.main = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, out_dim, kernel_size=4, stride=2, padding=1),  # Final output
            nn.Tanh()  # Final activation
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        return self.main(x)