from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic_Generator_1(nn.Module):
    def __init__(self, input_size = 5, output_size = 3):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_size, 128, 16, padding = 'same'),
            nn.ReLU()
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(128, 512, 3, padding = 'same'),
            nn.ReLU()
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(512, output_size, 3, padding = 'same'),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # B x 5 x 1 x 1
        x = self.block_1(x)
        # B x 64 x 16 x 16
        
        x = F.interpolate(
            input = x,
            scale_factor = 2,
            mode = "nearest"
        )
        # B x 64 x 32 x 32
        
        x = self.block_2(x)
        # B x 512 x 32 x 32
        
        x = F.interpolate(
            input = x,
            scale_factor = 2,
            mode = "nearest"
        )
        # B x 512 x 64 x 64
        
        x = self.block_3(x)
        # B x 3 x 64 x 64
        
        return x

class Basic_Discriminator_1(nn.Module):
    def __init__(self, input_size = 3, output_size = 2):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_size, 128, 3, padding = 'same'),
            nn.ReLU()
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(128, 512, 3, padding = 'same'),
            nn.ReLU()
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(512, output_size, 1, padding = 'same'),
            #nn.ReLU()
        )
    
    def forward(self, x):
        # B x 3 x 64 x 64
        
        x = F.interpolate(
            input = x,
            scale_factor = 0.5,
            mode = "nearest"
        )
        # B x 3 x 32 x 32
        
        x = self.block_1(x)
        # B x 128 x 32 x 32
        
        x = F.interpolate(
            input = x,
            scale_factor = 0.5,
            mode = "nearest"
        )
        # B x 128 x 16 x 16
        
        x = self.block_2(x)
        # B x 512 x 16 x 16
        
        x = F.interpolate(
            input = x,
            size = (1, 1),
            mode = "nearest"
        )
        # B x 512 x 1 x 1
        
        x = self.block_3(x)
        # B x 2 x 1 x 1
        
        return x