import torch
import pandas as pd
import utils
from pickle import dump
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

test_batch_size = 128

dataset = 'WUSTL-IIoT'
normal_data = pd.read_csv('Pre-Process/fe80dacb8afffe08ed2a_dataset.csv')
test_data = normal_data[int(len(normal_data)*.7):]
test_data = normal_data
train_label = test_data['Traffic']

model_name = 'model_6'
model_path = f'models/{dataset}/{model_name}/'
model, data_info = utils.load_model(model_path, device)

assert(data_info['dataset'] == dataset)


test_result = utils.prepare_data(test_data, dataset, model_path)
assert(data_info['data_columns'] == list(test_result.columns))





if __name__ == '__main__':

    test_loader = utils.get_dataLoader(test_result,train_label, 5, device, batch_size = test_batch_size)
    accuracy, avg_loss, predictions, true_labels = model.test_model(test_loader)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Sample Predictions: {predictions[:10]}")
    print(f"Sample True Labels: {true_labels[:10]}")