import torch 
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, context:torch.Tensor,time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if(isinstance(layer,UNET_AttentionBlock)):
                x = layer(x,context)
            elif isinstance(layer, UNET_residualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x



class TimeEmbedding(nn.Module):
    def __init__(self,n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd,n_embd*4)
        self.linear_2 = nn.Linear(n_embd*4,n_embd*4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.linear_2(F.silu(x))
        # (1,320) -> (1,1280)
        return x

class UpSample(nn.Module):
    def __init__(self,channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding =1)

    def forward(self,x):
        # (Batch_Size, Features,Hieght,Width) -> (Batch_Size, Features,Hieght*2,Width*2)
        x = F.interpolate(x,scale_factor=2,mode = "nearest")
        return self.conv(x)

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size =3, padding =1)

    def forward(self,x):
        #x: (Batch_Size,320,Hieght//8,Width//8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        #x: (Batch_Size,4,Hieght//8,Width//8)
        return x

class UNET_residualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,n_time =1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size =3, padding =1)
        self.linear_time = nn.Linear(n_time,out_channels)

        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size =3, padding =1)

        if in_channels ==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer =  nn.Conv2d(in_channels, out_channels, kernel_size =1, padding =0)

    def forward(self, feature, time):
        # feature: (Batch_size, In_channels, Hieght,Width)
        # time: (1,1280)
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_merged(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature +time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int, n_embd:int, d_context =768):
        super().__init__()
        channels = n_head * n_embd
        self.groupnorm = nn.GroupNorm(32,channels,eps = 1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head,channels,in_proj_bias =False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head,channels,d_context,in_proj_bias =False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)
        self.conv_output = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x, context):
        # Inputs:
        # latent Image: (Batch_Size, 4, Hieght//8, Width//8)
        # Prompt context from attention layer: (Batch_Size,Seq_Len, Dim)
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n,c,h,w = x.shape
        x = x.view((n,c,h*w))
        x = x.transpose(-1,-2) # rearranging the hieght and width
        #normalization and self attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        residue_short = x
        #normalization and cross attention with skip connection
        x = self.layernorm_2(x)
        # cross attention
        self.attention_2(x,context)
        x+=residue_short
        residue_short = x
        #normalization with feed forward with GeGLU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2,dim = -1)
        x *= F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))
        return self.conv_output(x)+ residue_long



















class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size,4,Hieght//8,Width//8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size =3,padding =1)),

            SwitchSequential(UNET_residualBlock(320,320), UNET_AttentionBlock(8,40)),
            
            SwitchSequential(UNET_residualBlock(320,320), UNET_AttentionBlock(8,40)),
            # (Batch_Size,320,Hieght//8,Width//8) ->(Batch_Size,320,Hieght//16,Width//16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size =3,stride =2,padding=1)),

            SwitchSequential(UNET_residualBlock(320,320*2), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_residualBlock(320*2,320*2), UNET_AttentionBlock(8,80)),
             # (Batch_Size,640,Hieght//16,Width//16) -> (Batch_Size,640,Hieght//32,Width//32)
            SwitchSequential(nn.Conv2d(320*2,320*2,kernel_size =3,stride =2,padding=1)),  
           
            SwitchSequential(UNET_residualBlock(320*2,320*4), UNET_AttentionBlock(8,160)),
            
            SwitchSequential(UNET_residualBlock(320*4,320*4), UNET_AttentionBlock(8,160)),
            # (Batch_Size,1280,Hieght//32,Width//32) -> (Batch_Size,1280,Hieght//64,Width//64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size =3,stride =2,padding =1)),

            SwitchSequential(UNET_residualBlock(320*2,320*4))
            # (Batch_Size,1280,Hieght//64,Width//64) -> (Batch_Size,1280,Hieght//64,Width//64)
            SwitchSequential(UNET_residualBlock(320*2,320*4))
        ])
        self.bottleneck = SwitchSequential(
            UNET_residualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_residualBlock(1280,1280),
        )
        self.decoders = nn.ModuleList([
            # (Batch_Size,2560,Hieght//64,Width//64) -> (Batch_Size,1280,Hieght//64,Width//64)
            SwitchSequential(UNET_residualBlock(320*8,320*4)),

            SwitchSequential(UNET_residualBlock(320*8,320*4)),

            SwitchSequential(UNET_residualBlock(320*8,320*4), UpSample(1280)),

            SwitchSequential(UNET_residualBlock(320*8,320*4), UNET_AttentionBlock(8,160)),
            
            SwitchSequential(UNET_residualBlock(320*8,320*4), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_residualBlock(320*6,320*4), UNET_AttentionBlock(8,160),UpSample(1280)),

            SwitchSequential(UNET_residualBlock(320*6,320*2), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_residualBlock(320*4,320*2), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_residualBlock(320*3,320*2), UNET_AttentionBlock(8,80), UpSample(640)),
        
            SwitchSequential(UNET_residualBlock(320*3,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_residualBlock(320*2,320), UNET_AttentionBlock(8,80)),                  

            SwitchSequential(UNET_residualBlock(320*2,320), UNET_AttentionBlock(8,40))
        ])

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
    
    def forward(self,latent:torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # Inputs:
        # latent Image: (Batch_Size, 4, Hieght//8, Width//8)
        # Prompt context from attention layer: (Batch_Size,Seq_Len, Dim)
        # time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        # (Batch_Size, 4, Hieght//8, Width//8) -> (Batch_Size, 320, Hieght//8, Width//8)
        output = self.unet(latent, context, time)
        # (Batch_Size, 320, Hieght//8, Width//8) -> (Batch_Size, 4, Hieght//8, Width//8)
        output = self.final(output)
        return output





