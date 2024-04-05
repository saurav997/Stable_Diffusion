import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self,n_heads:int,d_embed:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed,3*d_embed,bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x:torch.Tensor,causal_mask=False):
        # x:(Batch_size, Seq_Len, Dim)
        input_shape = x.shape
        batch_size,Seq_Len,d_embed = input_shape
        interim_shape = (batch_size,Seq_Len,self.n_heads,self.d_head)
        # To get the querry,key and value matrices we divide output of in_proj into 3 chunks
        q, k ,v = self.in_proj(x).chunk(3,dim = -1)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        # From above the dimension of q,k and v are (Batch_size,number of Heads,Seq_len,dimension per head)
        # (Batch_Size, H, Seq_Len,Seq_len)
        weight = q@k.transpose(-1,-2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).trui(1)
            weight.masked_fill_(mask,-torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        output = weight@v
        # (Batch_Size, Seq_Len, H, Dim/H)
        output = output.transpose(1,2)
        output = self.out_proj(output.reshape(input_shape))
        
        return output



