import torch
import torch.nn as nn
import numpy as np

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc

    def forward(self, x):
        seq_len = x.size(1)
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.positional_encoding[:, :seq_len, :].to(x.device)
        return token_embeddings + position_embeddings

# Define parameters
vocab_size = 100  # Assume we have 100 different tokens
d_model = 16      # Dimensionality of the embeddings
max_seq_len = 10  # Maximum sequence length

# Instantiate the embedding layer
embedding_layer = TransformerEmbedding(vocab_size, d_model, max_seq_len)

# Create a random batch of sequences (numerical dataset)
batch_size = 5
input_sequences = torch.randint(0, vocab_size, (batch_size, max_seq_len))  # Batch of 5 sequences

# Apply the embedding layer to the input sequences
output = embedding_layer(input_sequences)

print("Input Sequences:")
print(input_sequences)
print("\nEmbedded Sequences with Positional Encoding:")
print(output)
