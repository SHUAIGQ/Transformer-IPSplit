import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .layers import Norm, EncoderLayer, get_clones
from .attention import PositionalEncoder,InputEmbeddings
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder (d_model, window, device)
        self.layers = get_clones(EncoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)
        
    def forward(self, src):
        x = self.pe(src)
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)
    

class Transformer(nn.Module):
    def __init__(self, d_model,  N_layers, attention, windows, device, dropout=0.5, d_ff=1024):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model, N_layers, attention, windows, device, dropout, d_ff)
        self.out = nn.Linear(d_model, 5).to(torch.device(device))
        self.seq = windows
        self.device = device


    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

     
    def forward(self, src):
        e_outputs = self.encoder(src)  # Only the encoder is used
        output = self.out(e_outputs)
        return output



    def train_model(self, data_loader, val_loader, epochs=1, print_every=1, return_evo=False):
        optim = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98))

        criterion = nn.CrossEntropyLoss()  


        train_losses_per_iteration = []
        val_losses_per_iteration = []
        train_losses_per_epoch = []
        val_losses_per_epoch = []
        train_accuracies = []
        val_accuracies = []


        for epoch in range(epochs):
            print(f'\nEpoch: {epoch + 1} of {epochs}')
            
            # Training Phase
            self.train()
            total_train_loss = 0
            correct_train_preds = 0
            total_train_samples = 0

            with tqdm(total=len(data_loader)) as pbar:
                for i, (input, labels) in enumerate(data_loader):
                    optim.zero_grad()

                    # Forward pass
                    preds = self.forward(input) # Forward pass
                    
                    # Flatten predictions and labels
                    model_output_flat = preds.view(-1, preds.size(-1))
                    trg_indices = torch.argmax(labels, dim=-1)
                    trg_flat = trg_indices.view(-1)
                     
                    # Compute loss
                    loss = criterion(model_output_flat, trg_flat)
                    total_train_loss += loss.item()
                    #train_losses.append(loss.item())
                    
                    # Backpropagation and optimization
                    loss.backward()
                    optim.step()
                    
                    # Calculate accuracy
                    _, predicted_classes = torch.max(preds, dim=-1)
                    correct_train_preds += (predicted_classes == trg_indices).sum().item()
                    total_train_samples += trg_indices.numel()
                                

                    # Store loss per iteration
                    if (i + 1) % print_every == 0:
                        train_losses_per_iteration.append(loss.item())
                        pbar.set_postfix({'Batch Loss': loss.item()})

                    pbar.update(1)

          
            avg_train_loss = total_train_loss / len(data_loader)
            train_losses_per_epoch.append(avg_train_loss)
            train_accuracy = correct_train_preds / total_train_samples
            train_accuracies.append(train_accuracy)
            

            train_accuracy = correct_train_preds / total_train_samples
            train_accuracies.append(train_accuracy)

            # Validation Phase
            self.eval()
            total_val_loss = 0
            correct_val_preds = 0
            total_val_samples = 0

            with torch.no_grad():
                with tqdm(total=len(val_loader)) as pbar:
                    for i, (input, labels) in enumerate(val_loader):
                        preds = self.forward(input)

                        # Flatten predictions and labels
                        model_output_flat = preds.view(-1, preds.size(-1))
                        trg_indices = torch.argmax(labels, dim=-1)
                        trg_flat = trg_indices.view(-1)

                        # Compute loss
                        loss = criterion(model_output_flat, trg_flat)
                        total_val_loss += loss.item()

                        # Calculate accuracy
                        _, predicted_classes = torch.max(preds, dim=-1)
                        correct_val_preds += (predicted_classes == trg_indices).sum().item()
                        total_val_samples += trg_indices.numel()



                        # Store loss per iteration
                        if (i + 1) % print_every == 0:
                            val_losses_per_iteration.append(loss.item())
                            pbar.set_postfix({'Validation Loss': loss.item()})
                    
                        pbar.update(1)


            avg_val_loss = total_val_loss / len(val_loader)
            val_losses_per_epoch.append(avg_val_loss)
            val_accuracy = correct_val_preds / total_val_samples
            val_accuracies.append(val_accuracy)

            # print(f"Epoch {epoch + 1}/{epochs}: Training Loss = {avg_Training_loss:.4f}, Training Accuracy = {train_accuracy:.4f}")
            # print(f"Epoch {epoch + 1}/{epochs}: Validation Loss = {avg_val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")
                
            
        
        return train_losses_per_iteration, val_losses_per_iteration
    

    def test_model(self, data_loader):
        self.eval()
        total_val_loss = 0
        correct_val_preds = 0
        total_val_samples = 0
        
        predictions = []
        true_labels = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
                with tqdm(total=len(data_loader)) as pbar:
                    for i, (input, labels) in enumerate(data_loader):
                        input = input.to(self.device)  # Move input to the correct device
                        labels = labels.to(self.device)  # Move labels to the correct device
                        
                        preds = self.forward(input)  # Model forward pass

                        # Flatten predictions and labels
                        model_output_flat = preds.view(-1, preds.size(-1))
                        trg_indices = torch.argmax(labels, dim=-1)
                        trg_flat = trg_indices.view(-1)

                        # Compute loss
                        loss = criterion(model_output_flat, trg_flat)
                        total_val_loss += loss.item()

                        # Calculate accuracy
                        _, predicted_classes = torch.max(preds, dim=-1)
                        correct_val_preds += (predicted_classes == trg_indices).sum().item()
                        total_val_samples += trg_indices.numel()
                        # Store predictions and true labels
                        predictions.extend(predicted_classes.cpu().tolist())
                        true_labels.extend(trg_indices.cpu().tolist())

                        print("model_output_flat:",model_output_flat)
                        print("trg_flat:",trg_flat)

                        pbar.update(1)

    # Compute final accuracy
        accuracy = correct_val_preds / total_val_samples if total_val_samples > 0 else 0.0
        avg_loss = total_val_loss / len(data_loader) if len(data_loader) > 0 else 0.0

        return accuracy, avg_loss, predictions, true_labels

          

         

      
