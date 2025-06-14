"""
data_preparation.py
-------------------
This module contains functions for data loading and preparation for the inertial odometry project.
It includes file list builders, data loaders, sequence splitters, and DataLoader creators.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import dataset functions from the dataset module
from dataset import load_oxiod_dataset, load_euroc_mav_dataset, load_dataset_6d_quat
import pickle
import os

class IMUDataNormalizer:
    """
    Normalizes IMU data (gyroscope and accelerometer) by standardizing to zero mean and unit variance.
    """
    def __init__(self):
        """Initialize the normalizer with default values."""
        self.gyro_mean = None
        self.gyro_std = None
        self.acc_mean = None
        self.acc_std = None
        self.is_fitted = False
    
    def fit(self, gyro_data, acc_data):
        """Compute normalization parameters from input data."""
        # Reshape to compute stats across all samples and time steps
        gyro_flat = gyro_data.reshape(-1, gyro_data.shape[-1])
        acc_flat = acc_data.reshape(-1, acc_data.shape[-1])
        
        # Compute mean and std for each channel
        self.gyro_mean = np.mean(gyro_flat, axis=0)
        self.gyro_std = np.std(gyro_flat, axis=0)
        self.acc_mean = np.mean(acc_flat, axis=0)
        self.acc_std = np.std(acc_flat, axis=0)
        
        # Handle near-zero standard deviations
        self.gyro_std = np.maximum(self.gyro_std, 1e-5)
        self.acc_std = np.maximum(self.acc_std, 1e-5)
        
        self.is_fitted = True
        print(f"Gyro mean: {self.gyro_mean}, std: {self.gyro_std}")
        print(f"Acc mean: {self.acc_mean}, std: {self.acc_std}")
    
    def transform(self, gyro_data, acc_data):
        """Apply normalization to input data."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        gyro_norm = (gyro_data - self.gyro_mean) / self.gyro_std
        acc_norm = (acc_data - self.acc_mean) / self.acc_std
        
        return gyro_norm, acc_norm
    
    def fit_transform(self, gyro_data, acc_data):
        """Fit normalizer to data and then transform it."""
        self.fit(gyro_data, acc_data)
        return self.transform(gyro_data, acc_data)
    
    def save(self, filepath):
        """Save the normalizer parameters to a file."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted normalizer")
        
        params = {
            'gyro_mean': self.gyro_mean,
            'gyro_std': self.gyro_std,
            'acc_mean': self.acc_mean,
            'acc_std': self.acc_std
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, filepath):
        """Load normalizer parameters from a file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.gyro_mean = params['gyro_mean']
        self.gyro_std = params['gyro_std']
        self.acc_mean = params['acc_mean']
        self.acc_std = params['acc_std']
        self.is_fitted = True

def build_file_lists(dataset_choice: str):
    """
    Build lists of file paths for IMU and ground truth data based on the chosen dataset.

    Parameters:
        dataset_choice (str): The dataset choice, either 'oxiod' for the Oxford dataset 
                              or any other value for the default dataset.

    Returns:
        tuple: A tuple containing two lists:
            - imu_files (list): List of file paths for IMU data.
            - gt_files (list): List of file paths for ground-truth data.
    """
    # Check if the dataset choice is 'oxiod' (Oxford Inertial Odometry Dataset)
    if dataset_choice == 'oxiod':
        # File paths for Oxford dataset
        imu_files = ['Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv']
        gt_files = ['Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv']
    else:
        # File paths for the default dataset (multiple sequences)
        imu_files = [
            'public_dataset/MH_01_easy/mav0/imu0/data.csv',
            'public_dataset/MH_03_medium/mav0/imu0/data.csv',
            'public_dataset/MH_04_difficult/mav0/imu0/data.csv',
            'public_dataset/V1_01_easy/mav0/imu0/data.csv',
            'public_dataset/V1_03_difficult/mav0/imu0/data.csv',
            'public_dataset/MH_02_easy/mav0/imu0/data.csv',
            'public_dataset/MH_05_difficult/mav0/imu0/data.csv',
            'public_dataset/V1_02_medium/mav0/imu0/data.csv'
        ]
        gt_files = [
            'public_dataset/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv',
            'public_dataset/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv'
        ]
    return imu_files, gt_files


def load_and_process_data(dataset_choice, imu_files, gt_files, 
                         window_size=200, stride=10, normalize=True):
    """Load and process data with optional normalization."""
    sequence_data = []
    sequence_lengths = []
    all_gyro_data = []
    all_acc_data = []

    # First pass: collect data for normalization
    if normalize:
        print("Collecting data for normalization...")
        for imu_file, gt_file in zip(imu_files, gt_files):
            if dataset_choice == 'oxiod':
                gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(imu_file, gt_file)
            else:
                gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(imu_file, gt_file)
            
            (x_gyro, x_acc), _, _, _ = load_dataset_6d_quat(
                gyro_data, acc_data, pos_data, ori_data,
                window_size=window_size, stride=stride
            )
            
            all_gyro_data.append(x_gyro)
            all_acc_data.append(x_acc)
    
    # Create and fit normalizer
    normalizer = None
    if normalize:
        print("Fitting normalizer...")
        combined_gyro = np.vstack(all_gyro_data)
        combined_acc = np.vstack(all_acc_data)
        normalizer = IMUDataNormalizer()
        normalizer.fit(combined_gyro, combined_acc)
        
        os.makedirs('models', exist_ok=True)
        normalizer.save('models/imu_normalizer.pkl')
    
    # Second pass: process data with normalization
    for imu_file, gt_file in zip(imu_files, gt_files):
        if dataset_choice == 'oxiod':
            gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(imu_file, gt_file)
        else:
            gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(imu_file, gt_file)

        (x_gyro, x_acc), (y_delta_p, y_delta_q), init_p, init_q = load_dataset_6d_quat(
            gyro_data, acc_data, pos_data, ori_data,
            window_size=window_size, stride=stride
        )
        
        # Apply normalization
        if normalize and normalizer is not None:
            x_gyro, x_acc = normalizer.transform(x_gyro, x_acc)
            print(f"Normalized data for {imu_file}")
        
        sequence_data.append({
            'x_gyro': x_gyro,
            'x_acc': x_acc,
            'y_delta_p': y_delta_p,
            'y_delta_q': y_delta_q,
            'init_p': init_p,
            'init_q': init_q
        })
        sequence_lengths.append(len(x_gyro))
    
    return sequence_data, sequence_lengths, normalizer


def split_sequences(sequence_data: list, train_ratio: float = 0.7) -> tuple:
    """
    Split sequences into training and validation sets.

    Parameters:
        sequence_data (list): List of sequence dictionaries.
        train_ratio (float): Ratio of sequences to be used for training (default is 0.7).

    Returns:
        tuple: A tuple containing two lists:
            - train_sequences (list): List of training sequence dictionaries.
            - val_sequences (list): List of validation sequence dictionaries.
    """
    num_sequences = len(sequence_data)
    # Determine the index to split the sequence list based on the training ratio
    split_idx = int(train_ratio * num_sequences)
    return sequence_data[:split_idx], sequence_data[split_idx:]


def concat_and_build_dataset(sequences: list) -> TensorDataset:
    """
    Concatenates sequence data from multiple sequences and builds a PyTorch TensorDataset.

    Args:
        sequences (list of dict): A list of dictionaries where each dictionary contains:
            - 'x_gyro' (np.ndarray): Gyroscope data.
            - 'x_acc' (np.ndarray): Accelerometer data.
            - 'y_delta_p' (np.ndarray): Delta P data.
            - 'y_delta_q' (np.ndarray): Delta Q data.
            (Note: Other keys such as 'init_p' and 'init_q' are ignored in this function.)

    Returns:
        TensorDataset: A PyTorch TensorDataset containing the concatenated tensors:
            - x_gyro (torch.FloatTensor): Concatenated gyroscope data.
            - x_acc (torch.FloatTensor): Concatenated accelerometer data.
            - y_delta_p (torch.FloatTensor): Concatenated delta P data.
            - y_delta_q (torch.FloatTensor): Concatenated delta Q data.
    """
    # Vertically stack data from each sequence (i.e., concatenate along the first dimension)
    x_gyro = np.vstack([seq['x_gyro'] for seq in sequences])
    x_acc = np.vstack([seq['x_acc'] for seq in sequences])
    y_delta_p = np.vstack([seq['y_delta_p'] for seq in sequences])
    y_delta_q = np.vstack([seq['y_delta_q'] for seq in sequences])

    # Convert the numpy arrays to PyTorch tensors
    x_gyro = torch.FloatTensor(x_gyro)
    x_acc = torch.FloatTensor(x_acc)
    y_delta_p = torch.FloatTensor(y_delta_p)
    y_delta_q = torch.FloatTensor(y_delta_q)

    # Create and return a TensorDataset from the tensors
    return TensorDataset(x_gyro, x_acc, y_delta_p, y_delta_q)


def create_dataloaders(train_dataset: TensorDataset, val_dataset: TensorDataset, batch_size: int = 32) -> tuple:
    """
    Creates DataLoader objects for the training and validation datasets.

    Parameters:
        train_dataset (TensorDataset): The training dataset.
        val_dataset (TensorDataset): The validation dataset.
        batch_size (int): The batch size for the DataLoader (default is 32).

    Returns:
        tuple: A tuple containing two DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
    """
    # Create a DataLoader for the training dataset (shuffle can be enabled if desired)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def quaternion_angle_error(q_true: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular difference (in radians) between two sets of quaternions.

    Parameters:
        q_true (torch.Tensor): The ground truth quaternion tensor of shape (N, 4).
        q_pred (torch.Tensor): The predicted quaternion tensor of shape (N, 4).

    Returns:
        torch.Tensor: A tensor containing the angular differences in radians for each quaternion pair.
    """
    # Normalize the predicted quaternions to ensure they are unit quaternions
    q_pred_norm = torch.nn.functional.normalize(q_pred, p=2, dim=-1)
    # Compute the dot product between true and normalized predicted quaternions for each pair
    dot = (q_true * q_pred_norm).sum(dim=-1)
    # Clamp the dot product to the valid range [-1, 1] to avoid numerical issues with arccos
    dot = torch.clamp(dot, -1.0, 1.0)
    # Calculate the angle error (in radians) by taking the arccos of the absolute dot product and doubling it
    angles = 2.0 * torch.acos(torch.abs(dot))
    return angles
