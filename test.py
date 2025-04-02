import torch
import pandas as pd
import utils
from pickle import dump
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

test_batch_size = 128

dataset = '30Nov'
model_name = 'model_1'
model_path = f'models/{dataset}/{model_name}/'
model, data_info = utils.load_model(model_path, device)
assert data_info['dataset'] == dataset

test_data = pd.read_pickle(f'data/{dataset}/test_30Nov.pkl')
test_result = utils.prepare_data(test_data, dataset, model_path)
assert data_info['data_columns'] == list(test_result.columns)


print(model)