import torch
import copy
from torch import nn
from .attention import SingleHeadAttention, MultiHeadAttention, GeneralAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, device, dropout, d_ff):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff).to(torch.device(device))
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.linear_2 = nn.Linear(d_ff, d_model).to(torch.device(device))
        self.device = device

    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, device, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size).to(torch.device(device)))
        self.bias = nn.Parameter(torch.zeros(self.size).to(torch.device(device)))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=(0,1), keepdim=True)
        x_std = x.std(dim=(0,1), keepdim=True)
        norm = self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention, device, dropout, d_ff):
        super().__init__()
        self.norm_1 = Norm(d_model, device).to(torch.device(device))
        self.norm_2 = Norm(d_model, device).to(torch.device(device))

        if attention == 'general':
            self.attn = GeneralAttention(d_model, device).to(torch.device(device))
        elif attention == 'single':
            self.attn = SingleHeadAttention(d_model, device).to(torch.device(device))
        elif isinstance(attention, int):
            heads = attention
            self.attn = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
        else:
            raise ValueError('Attention type not recognized')

        self.ff = FeedForward(d_model, device, dropout, d_ff).to(torch.device(device))
        self.dropout_1 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_2 = nn.Dropout(dropout).to(torch.device(device))

    def forward(self, x):
        x_norm_1 = self.norm_1(x)
        x_dropout_1 = x + self.dropout_1(self.attn(x_norm_1, x_norm_1, x_norm_1))
        x_norm_2 = self.norm_2(x_dropout_1)
        x_dropout_2 = x_dropout_1 + self.dropout_2(self.ff(x_norm_2))
        return x_dropout_2



def get_clones(module, N_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N_layers)])
