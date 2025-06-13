"""
model.py
--------
This module defines neural network architectures (LSTM, TCN, Transformer) for inertial odometry,
along with custom loss functions. It also includes a helper function to instantiate a model by type.

Reference: https://github.com/jpsml/6-DOF-Inertial-Odometry/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Tuple

# TODO

# ----------------------------
# Loss Functions
# ----------------------------

class QuaternionLoss:
    """
    A collection of static methods to compute errors between quaternions.
    """

    @staticmethod
    def quaternion_phi_3_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the φ₃ error between two quaternions.
        Use this if you want to measure the angular (geodesic) difference between the quaternions.

        This error is calculated as the arccosine of the absolute dot product between the ground truth
        and the normalized predicted quaternion.

        Args:
            y_true: Ground truth quaternion tensor.
            y_pred: Predicted quaternion tensor.

        Returns:
            Tensor representing the φ₃ error.
        """
        # Normalize the predicted quaternion to ensure unit norm.
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        # Compute the dot product along the last dimension.
        dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
        # Return the angle error.
        return torch.acos(torch.abs(dot_product))

    @staticmethod
    def quaternion_phi_4_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the φ₄ error between two quaternions.

        The error is defined as one minus the absolute dot product between the ground truth and the
        normalized predicted quaternion.

        Args:
            y_true: Ground truth quaternion tensor.
            y_pred: Predicted quaternion tensor.

        Returns:
            Tensor representing the φ₄ error.
        """
        # Normalize predicted quaternion.
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        # Compute dot product and derive error.
        dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
        return 1 - torch.abs(dot_product)

    @staticmethod
    def quaternion_mean_multiplicative_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute a simplified quaternion multiplicative error using Mean Squared Error (MSE)
        between the ground truth and the normalized predicted quaternion.

        Args:
            y_true: Ground truth quaternion tensor.
            y_pred: Predicted quaternion tensor.

        Returns:
            Tensor representing the MSE error.
        """
        # Normalize predicted quaternion.
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        # Compute MSE loss.
        return F.mse_loss(y_true, y_pred_normalized)

class CustomMultiLossLayer(nn.Module):
    """
    A custom multi-loss layer that automatically weights multiple loss components.

    The layer maintains learnable log variance parameters for each output loss, which are used
    to balance the different loss terms during training.
    """
    def __init__(self, nb_outputs: int = 2):
        """
        Initialize the multi-loss layer.

        Args:
            nb_outputs: Number of outputs (loss components) to weight.
        """
        super().__init__()
        self.nb_outputs = nb_outputs
        # Initialize log variance parameters for each loss component.
        self.log_vars = nn.Parameter(torch.zeros(nb_outputs))
        
    def forward(self, y_true_list: List[torch.Tensor], y_pred_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the weighted sum of multiple loss components.

        For the first output, the L1 loss is used, and for the others, a quaternion multiplicative loss is used.
        Each loss term is scaled by an automatically learned weight.

        Args:
            y_true_list: List of ground truth tensors.
            y_pred_list: List of predicted tensors.

        Returns:
            Averaged weighted loss.
        """
        loss = 0
        # Iterate over each loss component.
        for i in range(self.nb_outputs):
            # Compute precision from log variance.
            precision = torch.exp(-self.log_vars[i])
            if i == 0:
                # For the first output, use L1 loss (mean absolute error)
                loss += precision * F.l1_loss(y_pred_list[i], y_true_list[i]) + self.log_vars[i]
            else:
                # For other outputs, use quaternion multiplicative error.
                loss += precision * QuaternionLoss.quaternion_mean_multiplicative_error(
                    y_true_list[i], y_pred_list[i]) + self.log_vars[i]
        # Return the mean loss.
        return loss.mean()

# ----------------------------
# Model Hyperparameters
# ----------------------------

# Hyperparameters for the LSTM-based architecture.
LSTM_PROPERTIES = {
    'hidden_size': 128,
    'dropout': 0.5,
    'kernel_size': 11,
    'num_layers': 2,
    'batch_first': True,
    'bidirectional': True,
}

# Hyperparameters for the TCN block.
TCNBLOCK_PROPERTIES = {
    'kernel_size': 7,
    'num_layers': 6,
    'dropout': 0.2,
    'activation': "gelu",
    'norm': "group",
}

# ----------------------------
# LSTM Model
# ----------------------------

class LSTMModel(nn.Module):
    """
    LSTM-based architecture for inertial odometry.

    This model processes gyroscope and accelerometer data through separate convolutional blocks,
    concatenates the features, and then uses two LSTM layers followed by fully-connected layers
    to predict position and orientation (quaternion).
    """
    def __init__(self):
        """
        Initialize the LSTMModel architecture.
        """
        super().__init__()
        lstm_props = LSTM_PROPERTIES

        # Convolutional blocks for gyroscope data.
        self.convA1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
        self.convA2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
        # Convolutional blocks for accelerometer data.
        self.convB1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
        self.convB2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
        # Pooling layer to reduce temporal dimensions.
        self.pool = nn.MaxPool1d(3)
        
        lstm_hidden = lstm_props['hidden_size']
        # First LSTM layer; input dimension is 256 due to feature concatenation.
        self.lstm1 = nn.LSTM(256, lstm_hidden, bidirectional=lstm_props['bidirectional'],
                             batch_first=lstm_props['batch_first'])
        # Second LSTM layer; input dimension doubles due to bidirectionality.
        self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden, bidirectional=lstm_props['bidirectional'],
                             batch_first=lstm_props['batch_first'])
        
        self.dropout = nn.Dropout(lstm_props['dropout'])
        # Fully connected layer for position output.
        self.fc_pos = nn.Linear(lstm_hidden * 2, 3)
        # Fully connected layer for quaternion output.
        self.fc_quat = nn.Linear(lstm_hidden * 2, 4)
    
    def forward(self, x_gyro: torch.Tensor, x_acc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTMModel.

        Args:
            x_gyro: Gyroscope data tensor of shape (batch, sequence_length, channels).
            x_acc: Accelerometer data tensor of shape (batch, sequence_length, channels).

        Returns:
            Tuple containing:
                - pos: Predicted position tensor.
                - quat: Normalized predicted quaternion tensor.
        """
        # Transpose inputs to (batch, channels, sequence_length) for convolution.
        x_gyro = x_gyro.transpose(1, 2)
        x_acc = x_acc.transpose(1, 2)
        
        # Process gyroscope data through convolutional layers.
        xa = F.relu(self.convA1(x_gyro))
        xa = F.relu(self.convA2(xa))
        xa = self.pool(xa)
        
        # Process accelerometer data through convolutional layers.
        xb = F.relu(self.convB1(x_acc))
        xb = F.relu(self.convB2(xb))
        xb = self.pool(xb)
        
        # Concatenate features from both sensors along the channel dimension.
        x = torch.cat([xa, xb], dim=1).transpose(1, 2)
        
        # Pass through the first LSTM layer.
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        # Pass through the second LSTM layer.
        x, _ = self.lstm2(x)
        # Use the output from the last time-step.
        x = self.dropout(x[:, -1, :])
        
        # Predict position.
        pos = self.fc_pos(x)
        # Predict and normalize quaternion.
        quat = F.normalize(self.fc_quat(x), p=2, dim=-1)
        return pos, quat

# ----------------------------
# TCN Model and Supporting Blocks
# ----------------------------

def make_norm(num_channels: int, norm_type: str = "group", num_groups: int = 8) -> nn.Module:
    """
    Return a normalization layer based on the specified type.

    Args:
        num_channels: Number of channels in the input.
        norm_type: Type of normalization ('group', 'layer', or 'batch').
        num_groups: Number of groups for GroupNorm.

    Returns:
        An instance of a normalization layer.
    """
    if norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type == "layer":
        return nn.LayerNorm(num_channels)
    else:
        return nn.BatchNorm1d(num_channels)

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) block.

    This block consists of two convolutional layers with normalization, activation, and dropout,
    and includes a residual connection. If the input and output channels differ, a downsampling
    layer is applied to the input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                 norm: str = 'group', activation: nn.Module = None):
        """
        Initialize the TCN block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride for the convolution.
            dilation: Dilation factor.
            norm: Type of normalization ('group', 'layer', or 'batch').
            activation: Activation function to use. If None, it will be selected based on global properties.
        """
        super().__init__()
        tcn_props = TCNBLOCK_PROPERTIES
        kernel_size = tcn_props['kernel_size']
        dropout = tcn_props['dropout']

        # Select activation function based on hyperparameters.
        if tcn_props['activation'] == "gelu":
            activation = nn.GELU()
        elif tcn_props['activation'] == "relu":
            activation = nn.ReLU()

        # Compute padding to maintain sequence length.
        padding = (kernel_size - 1) * dilation

        # First convolutional layer with weight normalization.
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding, dilation=dilation))
        self.norm1 = nn.GroupNorm(8, out_channels) if norm == 'group' else nn.BatchNorm1d(out_channels)
        self.act1 = activation

        # Second convolutional layer.
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding, dilation=dilation))
        self.norm2 = nn.GroupNorm(8, out_channels) if norm == 'group' else nn.BatchNorm1d(out_channels)
        self.act2 = activation

        self.dropout = nn.Dropout(dropout)

        # Downsample input if dimensions do not match.
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels) if norm == 'group' else nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN block.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Output tensor with residual connection added.
        """
        # First convolutional block.
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)

        # Second convolutional block.
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)

        # Apply residual connection.
        res = x if self.downsample is None else self.downsample(x)
        # Adjust sequence length if necessary.
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]
        return out + res

class TCNModel(nn.Module):
    """
    TCN-based model for inertial odometry.

    This model processes gyroscope and accelerometer data through separate TCN branches,
    merges their features, and then predicts position and orientation via fully-connected layers.
    """
    def __init__(self):
        """
        Initialize the TCNModel architecture.
        """
        super().__init__()
        tcn_props = TCNBLOCK_PROPERTIES

        # Select activation function based on hyperparameters.
        if tcn_props['activation'] == "gelu":
            activation = nn.GELU()
        elif tcn_props['activation'] == "relu":
            activation = nn.ReLU()

        # Build TCN branch for gyroscope data.
        self.gyro_tcn = nn.Sequential(
            TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'], activation=activation),
            TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
            TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32, norm=tcn_props['norm'], activation=activation),
        )

        # Build TCN branch for accelerometer data.
        self.acc_tcn = nn.Sequential(
            TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'], activation=activation),
            TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
            TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32, norm=tcn_props['norm'], activation=activation),
        )

        # Feature processing layer to combine information from both sensors.
        self.feature_processing = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.GroupNorm(8, 512),
            activation,
            nn.Dropout(tcn_props['dropout']),
        )

        # Global average pooling to reduce temporal dimension.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected path for predicting position.
        self.pos_path = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Dropout(tcn_props['dropout']),
            nn.Linear(256, 3),
        )
        # Fully connected path for predicting quaternion.
        self.quat_path = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x_gyro: torch.Tensor, x_acc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TCNModel.

        Args:
            x_gyro: Gyroscope data tensor of shape (batch, sequence_length, channels).
            x_acc: Accelerometer data tensor of shape (batch, sequence_length, channels).

        Returns:
            Tuple containing:
                - pos: Predicted position tensor.
                - quat: Normalized predicted quaternion tensor.
        """
        # Transpose inputs to (batch, channels, sequence_length) for convolution.
        x_gyro = x_gyro.transpose(1, 2)
        x_acc = x_acc.transpose(1, 2)

        # Process data through respective TCN branches.
        x_gyro = self.gyro_tcn(x_gyro)
        x_acc = self.acc_tcn(x_acc)

        # Concatenate features from both sensors.
        x = torch.cat([x_gyro, x_acc], dim=1)
        x = self.feature_processing(x)
        # Apply global average pooling to collapse the temporal dimension.
        x = self.global_pool(x).squeeze(-1)

        # Predict position.
        pos = self.pos_path(x)
        # Predict quaternion and normalize to unit length.
        quat = self.quat_path(x)
        quat = F.normalize(quat, p=2, dim=-1)
        return pos, quat

# ----------------------------
# Model Creation Helper
# ----------------------------

def create_model(model_type: str = 'lstm') -> nn.Module:
    """
    Instantiate a model based on the given type.

    Args:
        model_type: Type of model to instantiate. Supported values are 'lstm' and 'tcn'.
                    (Note: 'transformer' is mentioned in the module docstring but not implemented.)

    Returns:
        An instance of the selected neural network model.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    if model_type == 'lstm':
        return LSTMModel()
    elif model_type == 'tcn':
        return TCNModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
