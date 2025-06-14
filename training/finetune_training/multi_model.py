import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Tuple

LSTM_PROPERTIES = {
    'hidden_size': 256,
    'dropout': 0.5,
    'kernel_size': 11,
    'num_layers': 3,
    'batch_first': True,
    'bidirectional': True,
}

class MultiSensorLSTMModel(nn.Module):
    def __init__(self, num_sensors=4):
        super().__init__()
        lstm_props = LSTM_PROPERTIES
        self.num_sensors = num_sensors
        
        # Convolutional blocks for each sensor (gyroscope)
        self.gyro_convs = nn.ModuleList()
        for i in range(num_sensors):
            sensor_convs = nn.ModuleDict({
                'conv1': nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size']),
                'bn1': nn.BatchNorm1d(128),
                'conv2': nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size']),
                'bn2': nn.BatchNorm1d(128)
            })
            self.gyro_convs.append(sensor_convs)
            
        # Convolutional blocks for each sensor (accelerometer)
        self.acc_convs = nn.ModuleList()
        for i in range(num_sensors):
            sensor_convs = nn.ModuleDict({
                'conv1': nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size']),
                'bn1': nn.BatchNorm1d(128),
                'conv2': nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size']),
                'bn2': nn.BatchNorm1d(128)
            })
            self.acc_convs.append(sensor_convs)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # Feature fusion layer (to combine all sensor features)
        self.fusion_conv = nn.Conv1d(num_sensors * 256, 512, kernel_size=1)
        self.fusion_bn = nn.BatchNorm1d(512)
        
        lstm_hidden = lstm_props['hidden_size']
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, lstm_hidden, bidirectional=lstm_props['bidirectional'],
                             batch_first=lstm_props['batch_first'])
        self.bn_lstm1 = nn.BatchNorm1d(lstm_hidden * 2)
        
        self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden, bidirectional=lstm_props['bidirectional'],
                             batch_first=lstm_props['batch_first'])
        self.bn_lstm2 = nn.BatchNorm1d(lstm_hidden * 2)
        
        # Third LSTM layer
        self.lstm3 = nn.LSTM(lstm_hidden * 2, lstm_hidden, bidirectional=lstm_props['bidirectional'],
                             batch_first=lstm_props['batch_first'])
        self.bn_lstm3 = nn.BatchNorm1d(lstm_hidden * 2)
        
        self.dropout = nn.Dropout(lstm_props['dropout'])
        
        # Output path
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.bn_fc1 = nn.BatchNorm1d(lstm_hidden)
        self.act = nn.GELU()
        
        # Separate output heads
        self.fc_pos = nn.Linear(lstm_hidden, 3)
        self.fc_quat = nn.Linear(lstm_hidden, 4)
    
    def forward(self, gyro_list, acc_list):
        """
        Forward pass for multiple IMU sensors
        
        Args:
            gyro_list: List of gyroscope tensors, each of shape [batch, seq_len, 3]
            acc_list: List of accelerometer tensors, each of shape [batch, seq_len, 3]
            
        Returns:
            Tuple of (position_delta, orientation_delta)
        """
        batch_size = gyro_list[0].shape[0]
        
        # Process each sensor's gyroscope data
        gyro_features = []
        for i, gyro in enumerate(gyro_list):
            # Transpose for convolution [batch, channels, seq_len]
            x = gyro.transpose(1, 2)
            x = F.gelu(self.gyro_convs[i]['bn1'](self.gyro_convs[i]['conv1'](x)))
            x = F.gelu(self.gyro_convs[i]['bn2'](self.gyro_convs[i]['conv2'](x)))
            x = self.pool(x)
            gyro_features.append(x)
            
        # Process each sensor's accelerometer data
        acc_features = []
        for i, acc in enumerate(acc_list):
            # Transpose for convolution [batch, channels, seq_len]
            x = acc.transpose(1, 2)
            x = F.gelu(self.acc_convs[i]['bn1'](self.acc_convs[i]['conv1'](x)))
            x = F.gelu(self.acc_convs[i]['bn2'](self.acc_convs[i]['conv2'](x)))
            x = self.pool(x)
            acc_features.append(x)
            
        # Concatenate features from all sensors
        all_features = gyro_features + acc_features
        x = torch.cat(all_features, dim=1)
        
        # Apply fusion layer to combine all sensor features
        x = F.gelu(self.fusion_bn(self.fusion_conv(x)))
        
        # Convert back to sequence format for LSTM
        x = x.transpose(1, 2)
        
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        x = self.bn_lstm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = x.transpose(1, 2)
        x = self.bn_lstm2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        # Third LSTM layer
        x, _ = self.lstm3(x)
        
        # Use last time step output
        x = x[:, -1, :]
        
        # Final batch normalization
        x = self.bn_lstm3(x.unsqueeze(2)).squeeze(2)
        x = self.dropout(x)
        
        # Shared feature extraction
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Output heads
        pos = self.fc_pos(x)
        quat = F.normalize(self.fc_quat(x), p=2, dim=-1)
        
        return pos, quat    
