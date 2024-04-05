import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1,channels)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x
        n,c,h,w = x.shape
        # (Batch_size,Channel,Hieght,Width)->(Batch_size,Channel,Hieght*Width)
        x = x.view(n,c,h*w)
        # (Batch_size,Channel,Hieght*Width)->(Batch_size,Hieght*Width,Channel)
        x = x.transpose(-1,-2)
        # (Batch_size,Hieght*Width,Channel)->(Batch_size,Hieght*Width,Channel)
        x = self.attention(x)
        # (Batch_size,Hieght*Width,Channel)->(Batch_size,Channel,Hieght*Width)
        x = x.transpose(-1,-2)
        x = x.view(n,c,h,w)
        #back to original dimension
        return x + residue


        


class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super.__init__()
        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size = 3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size = 3, padding=1)
        if in_channels ==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding = 0)
    
    def forward(self,x:torch.tensor) ->torch.Tensor:
        # x: (Batch_size,In_channels,Hieght,Width)
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)



class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1 padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding =1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            #Image size remains the same till here i.e. (Batch_size,512,Hieght/8,Width/8)
            # (Batch_size,512,Hieght/8,Width/8) -> (Batch_size,512,Hieght/4,Width/4)
            nn.Upscale(scale_factor = 2),
            nn.Conv2d(512,512,kernel_size =3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            # (Batch_size,512,Hieght/4,Width/4) -> (Batch_size,512,Hieght/2,Width/2)
            nn.Upscale(scale_factor = 2),

            nn.Conv2d(512,512,kernel_size =3,padding=1),

            VAE_ResidualBlock(512,512//2),
            VAE_ResidualBlock(512//2,512//2),
            VAE_ResidualBlock(512//2,512//2),
            # (Batch_size,256,Hieght,Width)
            nn.Upscale(scale_factor = 2),
            nn.Conv2d(256,256,kernel_size =3,padding=1),

            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128),
            nn.SiLU(),
            # (Batch_size,3,Hieght,Width)
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x /=0.18215
        for module in self:
            x = module(x)
        return x
        
