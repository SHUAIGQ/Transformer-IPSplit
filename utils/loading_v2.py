import torch
import torch as nn
import numpy as np
import pandas as pd
import os
import pickle
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from .transformer import Transformer
from torch.utils.data import Dataset

class seq_Loader(Dataset):
    def __init__(self, init_data, seq, num_classes,device):
        super(seq_Loader, self).__init__() 

        self.dataset = init_data
        self.seq = seq
        self.device = device
        self.features = init_data.iloc[:, :-1]  # All columns except the last one
        self.labels = init_data.iloc[:, -1:]
        self.num_classes = num_classes

        self.features = torch.from_numpy(np.array(self.features)).float().to(torch.device(self.device)) # All columns except the last one
        self.labels = torch.from_numpy(np.array(self.labels)).long().to(torch.device(self.device))  # The last column 
        self.n_samples =  self.dataset.shape[0]

    
    def __len__(self):
           return self.n_samples
    
    def __getitem__(self, idx):
        feature = self.features[idx:idx+self.seq]  
        label = self.labels[idx:idx+self.seq]      

        label = F.one_hot(label, num_classes=self.num_classes).float()
        label = label.squeeze(-1)

        return feature, label

    
def get_dataLoader(data, window_size,num_classes, device, batch_size = 128):
    train_set = seq_Loader(data, window_size,num_classes, device)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False 
    )
   
    return train_loader

def val_dataLoader(data, window_size, device, batch_size = 1):
    test_set = seq_Loader(data, window_size, device)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False 
    )
    return test_loader

def prepare_data(data, dataset, scaling='gaussian', columns='normal'):
    print("Column names in the DataFrame:", data.columns)
    if dataset == 'UNSW-NB15':
        pred_columns = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
                        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                        'dintpkt', 'tcprtt', 'synack', 'ackdat']
        extra_columns = ['is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
                         'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
                         'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'trans_depth', 'res_bdy_len']
        min_columns = ['dur', 'sbytes', 'dbytes', 'sload', 'dload']
        if columns == 'normal':
            columned_data = data[pred_columns]
        elif columns == 'minimal':
            columned_data = data[min_columns]
        elif columns == 'extended':
            columned_data = data[pred_columns + extra_columns]
        else:
            columned_data = data

    elif dataset == 'WUSTL-IIoT':
        pred_columns = ['Mean', 'Sport', 'Dport', 'SrcPkts', 'DstPkts', 'TotPkts', 'DstBytes', 'SrcBytes',
                        'TotBytes', 'SrcLoad', 'DstLoad', 'Load', 'SrcRate', 'DstRate', 'Rate', 'SrcLoss',
                        'DstLoss', 'Loss', 'pLoss', 'SrcJitter', 'DstJitter', 'SIntPkt', 'DIntPkt', 'Proto',
                        'Dur', 'TcpRtt', 'IdleTime', 'Sum', 'Min', 'Max', 'sDSb', 'sTtl', 'dTtl', 'sIpId',
                        'dIpId', 'SAppBytes', 'DAppBytes', 'TotAppByte', 'SynAck', 'RunTime', 'sTos', 'SrcJitAct',
                        'DstJitAct']
        columned_data = data[pred_columns]
    elif dataset == 'combined_v2':
        pred_columns =['Mean','SrcRate','DstRate','Rate','Dur','Sum','Min','Max','TotAppByte','SynAck']
        columned_data = data[pred_columns]
    elif dataset == 'Combined_data_final':
        pred_columns =['Mean','SrcRate','DstRate','Rate','Dur','Sum','Min','Max','TotAppByte','SynAck']
        columned_data = data[pred_columns]
    elif dataset == 'combined':
        pred_columns = ['Mean', 'Sport', 'Dport', 'SrcPkts', 'DstPkts', 'TotPkts', 'DstBytes', 'SrcBytes',
                        'TotBytes', 'SrcLoad', 'DstLoad', 'Load', 'SrcRate', 'DstRate', 'Rate', 'SrcLoss',
                        'DstLoss', 'Loss', 'pLoss', 'SrcJitter', 'DstJitter', 'SIntPkt', 'DIntPkt', 'Proto',
                        'Dur', 'TcpRtt', 'IdleTime', 'Sum', 'Min', 'Max', 'sDSb', 'sTtl', 'dTtl', 'sIpId',
                        'dIpId', 'SAppBytes', 'DAppBytes', 'TotAppByte', 'SynAck', 'RunTime', 'sTos', 'SrcJitAct',
                        'DstJitAct']
        columned_data = data[pred_columns]

    elif dataset == 'iot23':
        pred_columns = ['MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance', 'H_L0.1_weight',
                        'H_L0.1_mean', 'H_L0.1_variance', 'HH_L0.1_weight', 'HH_L0.1_mean', 'HH_L0.1_std',
                        'HH_L0.1_magnitude', 'HH_L0.1_radius', 'HH_L0.1_covariance', 'HH_L0.1_pcc',
                        'HH_jit_L0.1_weight', 'HH_jit_L0.1_mean', 'HH_jit_L0.1_variance', 'HpHp_L0.1_weight',
                        'HpHp_L0.1_mean', 'HpHp_L0.1_std', 'HpHp_L0.1_magnitude', 'HpHp_L0.1_radius',
                        'HpHp_L0.1_covariance', 'HpHp_L0.1_pcc', 'label']
        columned_data = data[pred_columns]
        
    elif dataset == '7Dec':
        # Columns for the CICIoT dataset
        pred_columns = ['flow_duration','header_length','duration','rate',
                        'srate','drate','fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number',
                        'ack_flag_number','ece_flag_number','cwr_flag_number','ack_count','syn_count','fin_count',
                        'urg_count','rst_count','http','https','dns','telnet','smtp','ssh','irc','tcp','udp','dhcp',
                        'arp','icmp','ipv','llc','tot_sum','min','max','avg','std','tot_size','iat','number','radius',
                        'covariance','variance','weight','label']
        columned_data = data[pred_columns]
        
    elif dataset == 'CICIoT':
        # Columns for the CICIoT dataset
        pred_columns = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
                        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
                        'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
                        'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP',
                        'ARP', 'ICMP', 'IPv', 'LLC', 'Sum', 'Min', 'Max', 'Mean', 'Std', 'Tot size', 'IAT', 'Number',
                        'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
        columned_data = data[pred_columns]
    else:
        raise ValueError('Dataset not implemented')

   
    if scaling == 'gaussian':
        scaler = StandardScaler()
        # Scaling only the features (not the label column)
        feature_data = columned_data.drop(columns=['label'])  # Ensure 'label' is excluded
        scaled_features = pd.DataFrame(scaler.fit_transform(feature_data.values), columns=feature_data.columns,
                                       index=feature_data.index)
        
        # Re-add the 'label' column without scaling
        scaled_data = pd.concat([scaled_features, columned_data['label']], axis=1)
        return scaled_data, scaler

    elif os.path.isfile(scaling + 'scaler.pkl'):
        scaler = pickle.load(open(scaling + 'scaler.pkl', 'rb'))
        feature_data = columned_data.drop(columns=['label'])
        scaled_features = pd.DataFrame(scaler.fit_transform(feature_data.values), columns=feature_data.columns,
                                       index=feature_data.index)
        scaled_data = pd.concat([scaled_features, columned_data['label']], axis=1)
        return scaled_data
    else:
        raise ValueError('Scaling method not implemented')

def load_model(model_path, device, return_info = True):
    model_info = pickle.load(open(model_path+'model_info.pkl', 'rb'))
    data_columns = model_info['data_columns']
    d_model = len(data_columns)
    attention = model_info['attention']
    if attention[:10] == 'multi-head':
        attention = int(attention.split(' ')[1])
    model = Transformer(d_model, model_info['N_layers'], attention, model_info['window_size'], device, model_info['dropout'], model_info['ff_neurons'])
    model.load_state_dict(torch.load(model_path+'model_state.pt', map_location=torch.device(device)))
    model_info = {
        'data_columns': data_columns,
        'window_size': model_info['window_size'],
        'batch_size': model_info['batch_size'],
        'dataset': model_info['dataset'],
        'dropout' : model_info['dropout'],
        'ff_neurons' : model_info['ff_neurons'] 
    }



    if return_info:
        return model, model_info
    else:
        return model
