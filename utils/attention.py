import torch
import math
from torch import nn


class InputEmbeddings(nn.Module):
    def _init_(self,d_model,window):
        super()._init_()
        self.d_model = d_model
        self.window = window
        self.embedding = nn.Embedding(window,d_model)

    def foward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, window, device, encoding_length = 10000, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.batch_first = batch_first
        pe = torch.zeros(window, d_model).to(torch.device(device))
        for pos in range(window):
            
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (encoding_length ** ((2 * i)/d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (encoding_length ** ((2 * (i + 1))/d_model))) 
        pe = pe.unsqueeze(0)
        self.pe = pe
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
def scaled_DPattention(q, k, v, d_attention, mask=None, dropout=None):
   
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_attention)
   
    # if mask is not None:
       
    #     scores = scores.masked_fill(mask == 0, -1e9)
    
    # scores = nn.functional.softmax(scores, dim=-1)
    # if dropout is not None:
    #     scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class GeneralAttention(nn.Module):
    def __init__(self, d_model, device, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        scores = scaled_DPattention(q, k, v, self.d_model, mask, self.dropout)   
        print(scores.size()) 
        output = scores
        return output
    
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, device, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.out = nn.Linear(d_model, d_model).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        q_linear = self.q_linear(q)
        k_linear = self.k_linear(k)
        v_linear = self.v_linear(v)
        scores = scaled_DPattention(q_linear, k_linear, v_linear, self.d_model, mask, self.dropout)
        output = self.out(scores)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.N_heads = heads

        self.d_head = (d_model // heads)+1
        self.q_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.v_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.k_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.out = nn.Linear(self.N_heads * self.d_head, d_model).to(torch.device(device))
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

     
        q_lin = self.q_linear(q)
        k_lin = self.k_linear(k)
        v_lin = self.v_linear(v)

        q_view = q_lin.view(batch_size, q_len, self.N_heads, self.d_head)
        k_view = k_lin.view(batch_size, k_len, self.N_heads, self.d_head)
        v_view = v_lin.view(batch_size, v_len, self.N_heads, self.d_head)
        
        
        q_heads = q_view.transpose(1,2)
        k_heads = k_view.transpose(1,2)
        v_heads = v_view.transpose(1,2)
        
        
        scores = scaled_DPattention(q_heads, k_heads, v_heads, self.d_head, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.N_heads * self.d_head)
        
        output = self.out(concat)
    
        return output