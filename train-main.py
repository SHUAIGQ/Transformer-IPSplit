import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder

device = torch.device("cuda")
print(f"Using {device} device")

dataset = 'WUSTL-IIoT'
normal_data = pd.read_csv('Pre-Process/192.168.0.44_dataset.csv')
train_df = normal_data[:int(len(normal_data)*.8)]
test_df = normal_data[int(len(normal_data)*.2):]


feature_data = train_df.drop(columns=['StartTime','LastTime','SrcAddr','DstAddr','Traffic','Target'])

#Train dataset
train_data, train_scaler = utils.prepare_data(feature_data, dataset)
dataset_length = len(train_data)
train_label = train_df['Traffic']






#Test dataset
test_feature_data = test_df.drop(columns=['StartTime','LastTime','SrcAddr','DstAddr','Traffic','Target'])
test_data, train_scaler = utils.prepare_data(test_feature_data, dataset)
dataset_length = len(train_data)
test_label = test_df['Traffic'] 



d_model = 43  # Reduced dimension model size
attention = 'single'
N_layers = 6
seq = 5
batch_size = 64
epochs = 10
dropout = 0.5
ff_neurons = 1024
num_classes = 5





 
# yhat = model.train_model(train_loader,  epochs=epochs, print_every=1, return_evo=True)
# print('Shape of yhat: ', yhat.shape) 

if __name__ == '__main__':

    train_loader = utils.get_dataLoader(train_data, train_label,seq, device, batch_size = batch_size)
    val_loader = utils.get_dataLoader(test_data,test_label, seq, device, batch_size = batch_size)

    model = utils.Transformer(d_model, N_layers, attention, seq, device, dropout, ff_neurons)
    model.initialize()

    
##-----------------Plot the training and validation loss per epoch----------------##


    train_losses, val_losses = model.train_model(train_loader, val_loader, epochs=epochs, print_every=1, return_evo=True) #validation loader,
     # Save losses
    df1 = pd.DataFrame({
        'train_loss': train_losses
    })
    df1.to_csv('train_losses.csv', index=False)

    df2 = pd.DataFrame({
        'train_loss': val_losses
    })
    df2.to_csv('val_losses.csv', index=False)



  


# #-----------------Plot the training and validation loss per iteration----------------##

#     train_losses_per_iteration, val_losses_per_iteration = model.train_model(
#         train_loader, val_loader, epochs=epochs, print_every=50, return_evo=True
#     )
#     # Save losses
#     df = pd.DataFrame({
#         'train_loss': train_losses_per_iteration
#     })
#     df.to_csv('losses_without_dropout.csv', index=False)

#     plt.plot(train_losses_per_iteration, label='Training Loss')
#    # plt.plot(val_losses_per_iteration, label='Validation Loss')
    
#     plt.title('Training and Validation Loss per Iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()


 

    # ##------------ Code to check data shape()------------------------------#
    # total=train_loader.__len__()
    # print(total)
    # for train_features, label in train_loader:
    #     input = train_features
    #     print(f"Feature batch shape: {train_features.size()}")
    #     print(f"input:{input}")
    #     print(f"label batch shape: {label.size()}")
    #     print(f"label:{label}")

    # ##-----------------------------------------------------------------------------#


    info_dict = {
        'data_columns' : list(train_data.columns),
        'attention' : attention,
        'N_layers' : N_layers,
        'seq' : seq,
        'batch_size' : batch_size,
        'model_info' : str(model),
        'epochs' : epochs,
        'dataset' : dataset,
        'dropout' : dropout,
        'ff_neurons' : ff_neurons,
        'num_classes' : num_classes,
        'd_model' : d_model
    }

    utils.save_model(model, info_dict, train_scaler)
