import torch
import torchvision
import torch as nn
import numpy as np
import pandas as pd
import os
import pickle
import utils
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader


device = torch.device("cuda")
print(f"Using {device} device")

dataset = '7Dec'
normal_data = pd.read_pickle('data/'+dataset+'/train_'+dataset+'.pkl')
train_data, train_scaler = utils.prepare_data(normal_data, dataset)
dataset_length = len(train_data)



d_model = 512  # Reduced dimension model size
attention = 'single'
N_layers = 4 
window_size = 47
batch_size = 128
epochs = 1
dropout = 0
ff_neurons = 1024
num_classes = 12

class seq_Loader(Dataset):
    def __init__(self, init_data, window, device):
        super(seq_Loader, self).__init__() 

        self.dataset = init_data
        self.window = window
        self.device = device
        self.features = init_data.iloc[:, :-1]  # All columns except the last one
        self.labels = init_data.iloc[:, -1:]
        

        self.features = torch.from_numpy(np.array(self.features)).float().to(torch.device(self.device)) # All columns except the last one
        self.labels = torch.from_numpy(np.array(self.labels)).float().to(torch.device(self.device))  # The last column 
        self.n_samples =  self.dataset.shape[0]
    
    def __len__(self):
           return (len(self.features) - (self.window - 1))
    

    
    def __getitem__(self, idx):
        
        return self.features[idx], self.labels[idx]



    
def get_dataLoader(data, window_size, device, batch_size = 128):
    train_set = seq_Loader(data, window_size, device)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False 
    )
   
    return train_loader

def val_dataLoader(data, window_size, device, batch_size = 1):
    test_set = seq_Loader(data, window_size, device)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False 
    )
    return test_loader


train_set = seq_Loader(train_data, window_size, device)


if __name__ == '__main__':
    features = torch.from_numpy(np.array(train_data.iloc[:, :-1])).float().to(torch.device(device))
    print (features)

    dataloader = DataLoader(dataset=train_set,batch_size=4,shuffle=True, num_workers=2)
    # for epoch in range(epochs):
    #     for i , (inputs,labels) in enumerate(dataloader):
    #         if (i+1) % 5 == 0:
    #             print(f'inputs: {inputs}')

    # for i , (inputs,labels) in enumerate(dataloader):
    #     print(f'inputs: {inputs}')

# # Backup loader 
# class seq_Loader(Dataset):
#     def __init__(self, init_data, window, device):
#         super(seq_Loader, self).__init__() 

#         self.dataset = init_data
#         self.window = window
#         self.device = device

#         self.features = init_data.iloc[:, :-1]  # All columns except the last one
#         self.labels = init_data.iloc[:, -1:]    # The last column 
    
#     def __len__(self):
#            return (len(self.features) - (self.window - 1))
    

    
#     def __getitem__(self, idx):
#         feature_window = self.features.iloc[idx:idx+self.window]
#         label_window = self.labels.iloc[idx:idx+self.window]
        
#         feature_tensor = torch.from_numpy(np.array(self.features)).float().to(torch.device(self.device))
#         label_tensor = torch.from_numpy(np.array(self.labels)).float().to(torch.device(self.device))
#         return feature_tensor, label_tensor