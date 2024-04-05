import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init(self):
        super.__init__(
            # (Batch_size,Channel,Hieght,Width)->(Batch_size,128,Hieght,Width),
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            # (Batch_size,128,Hieght,Width)->(Batch_size,128,Hieght,Width),
            VAE_ResidualBlock(128,128),
            # (Batch_size,128,Hieght,Width)->(Batch_size,128,Hieght,Width),
            VAE_ResidualBlock(128,128),
            # (Batch_size,128,Hieght,Width)->(Batch_size,128,Hieght//2,Width//2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            # (Batch_size,128,Hieght//2,Width//2)->(Batch_size,128*2,Hieght//2,Width//2),
            VAE_ResidualBlock(128,128*2),
            # (Batch_size,128*2,Hieght//2,Width//2)->(Batch_size,128*2,Hieght//2,Width//2),
            VAE_ResidualBlock(128*2,128*2),
            # (Batch_size,128*2,Hieght//2,Width//2)->(Batch_size,128*2,Hieght//4,Width//4),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            # (Batch_size,256,Hieght//4,Width//4)->(Batch_size,512,Hieght//4,Width//4)
            VAE_ResidualBlock(256,512),
            # (Batch_size,512,Hieght//4,Width//4)->(Batch_size,512,Hieght//4,Width//4)
            VAE_ResidualBlock(512,512),
            # (Batch_size,512,Hieght//4,Width//4)->(Batch_size,512,Hieght//8,Width//8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            # for the next 3 (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            VAE_AttentionBlock(512),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            VAE_ResidualBlock(512,512),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            nn.GroupNorm(32,512),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            nn.SiLU(),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,512,Hieght//8,Width//8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (Batch_size,512,Hieght//8,Width//8)->(Batch_size,8,Hieght//8,Width//8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            # (Batch_size,8,Hieght//8,Width//8)->(Batch_size,8,Hieght//8,Width//8)
            )

    def forward(self,x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size,Channel,Hieght,Width)
        # noise: (Batch_size,output_Channel,Hieght//8,Width//8)
        for module in self:
            if getattr(module , 'stride',None)==(2,2):
                #pad module input only on the bottom and right sides because 
                x = F.pad(x,(0,1,0,1))
            x = module(x)
        #getting the mean and variance out of the final output by cutting the x output into 2 chunks
        mean,log_variance = torch.chunk(x,2,dim=1)
        log_variance = torch.clamp(log_variance, -30,20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        # dimension(mean,stdev) = (Batch_size,4,Hieght//8,Width//8)
        # N(0,1) -> N(mean,variance) using x = mean +stdev*x
        x = mean +stdev*noise
        # output scaling
        x *= .18215
        return x
        