"""
finetune_train.py
-----------------
Fine-tune a pre-trained inertial odometry model on custom MEMS data.
Usage: python finetune_train.py <pretrained_model_path> --output <output_name>
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from pathlib import Path

# Import modules
from model import create_model, CustomMultiLossLayer
from data_preparation import IMUDataNormalizer, quaternion_angle_error

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
WINDOW_SIZE = 200
STRIDE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4


def load_finetune_dataset(window_size=200, prediction_span=10, stride=10, normalize=True):
    """
    Load and process the finetune dataset files with proper normalization.
    """
    print("Loading finetune dataset...")
    
    # Define file paths
    imu_file = 'finetune_dataset/MEMS_4_log_2_imu_1_processed.csv'
    gt_file = 'finetune_dataset/EKF_processed_interpolated.csv'
    
    # Check if files exist
    if not os.path.exists(imu_file):
        raise FileNotFoundError(f"IMU file not found: {imu_file}")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    print(f"Loading IMU data from {imu_file}")
    imu_data = pd.read_csv(imu_file)
    
    print(f"Loading ground truth data from {gt_file}")
    gt_data = pd.read_csv(gt_file)
    
    print(f"IMU data shape: {imu_data.shape}, Ground truth data shape: {gt_data.shape}")
    
    # Extract relevant columns
    gyro_data = imu_data[['gyro_1_x', 'gyro_1_y', 'gyro_1_z']].values
    acc_data = imu_data[['acc_1_x', 'acc_1_y', 'acc_1_z']].values
    
    # Extract position and orientation from ground truth
    pos_data = gt_data[['Displacement X (m)', 'Displacement Y (m)', 'Displacement Z (m)']].values
    ori_data = gt_data[['q0(w)', 'q1(x)', 'q2(y)', 'q3(z)']].values
    
    print(f"Data shapes - Gyro: {gyro_data.shape}, Acc: {acc_data.shape}")
    print(f"GT shapes - Pos: {pos_data.shape}, Ori: {ori_data.shape}")
    
    # Process data into windows with short-term prediction targets
    x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q = process_into_windows(
        gyro_data, acc_data, pos_data, ori_data,
        window_size=window_size, prediction_span=prediction_span, stride=stride
    )
    
    # Create normalizer for the data
    normalizer = None
    if normalize:
        normalizer = IMUDataNormalizer()
        normalizer.fit(x_gyro, x_acc)
        print("Fitted normalizer to finetune dataset")
        
        # Apply normalization
        x_gyro, x_acc = normalizer.transform(x_gyro, x_acc)
        print("Applied normalization to finetune dataset")
    
    # Convert to tensors
    x_gyro = torch.FloatTensor(x_gyro)
    x_acc = torch.FloatTensor(x_acc)
    y_delta_p = torch.FloatTensor(y_delta_p)
    y_delta_q = torch.FloatTensor(y_delta_q)
    
    print(f"Processed {len(x_gyro)} windows of data")
    
    return x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q, normalizer

def load_multi_sensor_dataset(sensor_files, gt_file, window_size=200, prediction_span=10, stride=10, normalize=True):
    """
    Load and process multiple IMU sensor files with proper normalization.
    """
    print(f"Loading multi-sensor dataset with {len(sensor_files)} sensors...")
    
    # Load ground truth data
    gt_data = pd.read_csv(gt_file)
    pos_data = gt_data[['Displacement X (m)', 'Displacement Y (m)', 'Displacement Z (m)']].values
    ori_data = gt_data[['q0(w)', 'q1(x)', 'q2(y)', 'q3(z)']].values
    
    # Load all IMU data files
    all_gyro_data = []
    all_acc_data = []
    normalizers = []
    
    for i, imu_file in enumerate(sensor_files):
        print(f"Loading sensor {i+1} data from {imu_file}")
        imu_data = pd.read_csv(imu_file)
        
        # Extract gyroscope and accelerometer data
        gyro_data = imu_data[[f'gyro_{i+1}_x', f'gyro_{i+1}_y', f'gyro_{i+1}_z']].values
        acc_data = imu_data[[f'acc_{i+1}_x', f'acc_{i+1}_y', f'acc_{i+1}_z']].values
        
        # Normalize if requested
        if normalize:
            normalizer = IMUDataNormalizer()
            gyro_data, acc_data = normalizer.fit_transform(gyro_data, acc_data)
            normalizers.append(normalizer)
        
        all_gyro_data.append(gyro_data)
        all_acc_data.append(acc_data)
    
    # Process each sensor's data into windows
    all_x_gyro = []
    all_x_acc = []
    
    for i in range(len(all_gyro_data)):
        # Process into windows but only keep the IMU data
        x_gyro, x_acc, _, _, _, _ = process_into_windows(
            all_gyro_data[i], all_acc_data[i], pos_data, ori_data, 
            window_size=window_size, prediction_span=prediction_span, stride=stride
        )
        all_x_gyro.append(x_gyro)
        all_x_acc.append(x_acc)
    
    # Process ground truth with any of the sensor data (they should be synchronized)
    _, _, y_delta_p, y_delta_q, init_p, init_q = process_into_windows(
        all_gyro_data[0], all_acc_data[0], pos_data, ori_data, 
        window_size=window_size, prediction_span=prediction_span, stride=stride
    )
    
    print(f"Processed {len(y_delta_p)} windows")
    
    return all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q, normalizers

def create_multi_sensor_train_val_split(all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q, val_ratio=0.2, batch_size=32, shuffle_train=True):
    """
    Create train/validation split for multi-sensor data, using the same data for both
    to enable overfitting testing.
    """
    # Convert all numpy arrays to tensors
    all_x_gyro_tensors = [torch.FloatTensor(x_gyro) for x_gyro in all_x_gyro]
    all_x_acc_tensors = [torch.FloatTensor(x_acc) for x_acc in all_x_acc]
    
    y_delta_p_tensor = torch.FloatTensor(y_delta_p)
    y_delta_q_tensor = torch.FloatTensor(y_delta_q)
    init_p_tensor = torch.FloatTensor(init_p)
    init_q_tensor = torch.FloatTensor(init_q)
    
    # Create custom dataset class for multiple sensors
    class MultiSensorDataset(torch.utils.data.Dataset):
        def __init__(self, gyro_list, acc_list, y_delta_p, y_delta_q, init_p, init_q):
            self.gyro_list = gyro_list
            self.acc_list = acc_list
            self.y_delta_p = y_delta_p
            self.y_delta_q = y_delta_q
            self.init_p = init_p
            self.init_q = init_q
            
        def __len__(self):
            return len(self.y_delta_p)
            
        def __getitem__(self, idx):
            gyro_data = [gyro[idx] for gyro in self.gyro_list]
            acc_data = [acc[idx] for acc in self.acc_list]
            return (
                gyro_data,
                acc_data,
                self.y_delta_p[idx],
                self.y_delta_q[idx],
                self.init_p[idx],
                self.init_q[idx]
            )
    
    # Create a single dataset with all data
    full_dataset = MultiSensorDataset(
        all_x_gyro_tensors, all_x_acc_tensors, 
        y_delta_p_tensor, y_delta_q_tensor, init_p_tensor, init_q_tensor
    )
    
    # Use the same dataset for both training and validation
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Using all {len(full_dataset)} samples for both training and validation")
    
    return train_loader, val_loader

def train_epoch_multi_sensor(model, dataloader, criterion, optimizer, device):
    """Run one epoch of training for multi-sensor model."""
    model.train()
    total_loss = 0
    total_pos_rmse = 0
    total_quat_angle = 0
    batch_count = 0
    
    for gyro_list, acc_list, y_delta_p, y_delta_q, _, _ in dataloader:
        # Move data to device
        gyro_list = [gyro.to(device) for gyro in gyro_list]
        acc_list = [acc.to(device) for acc in acc_list]
        y_delta_p = y_delta_p.to(device)
        y_delta_q = y_delta_q.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pos_pred, quat_pred = model(gyro_list, acc_list)
        
        # Calculate loss
        loss = criterion([pos_pred, quat_pred], [y_delta_p, y_delta_q])
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Position RMSE
            pos_rmse = torch.sqrt(torch.mean(torch.sum((pos_pred - y_delta_p) ** 2, dim=1)))
            
            # Quaternion angle error
            quat_error = quaternion_angle_error(y_delta_q, quat_pred)
            quat_angle = quat_error.mean() * 180.0 / np.pi  # Convert to degrees
        
        # Accumulate batch metrics
        batch_size = y_delta_p.size(0)
        total_loss += loss.item() * batch_size
        total_pos_rmse += pos_rmse.item() * batch_size
        total_quat_angle += quat_angle.item() * batch_size
        batch_count += batch_size
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_pos_rmse = total_pos_rmse / batch_count
    avg_quat_angle = total_quat_angle / batch_count
    
    return avg_loss, avg_pos_rmse, avg_quat_angle

def validate_multi_sensor(model, dataloader, criterion, device):
    """Validate the multi-sensor model."""
    model.eval()
    val_losses = []
    pos_errors = []
    quat_angles = []
    
    with torch.no_grad():
        for gyro_list, acc_list, y_delta_p, y_delta_q, _, _ in dataloader:
            # Move data to device
            gyro_list = [gyro.to(device) for gyro in gyro_list]
            acc_list = [acc.to(device) for acc in acc_list]
            y_delta_p = y_delta_p.to(device)
            y_delta_q = y_delta_q.to(device)
            
            # Forward pass
            pred_delta_p, pred_delta_q = model(gyro_list, acc_list)
            
            # Compute loss
            loss = criterion([pred_delta_p, pred_delta_q], [y_delta_p, y_delta_q])
            val_losses.append(loss.item())
            
            # Compute position error
            pos_error = torch.norm(pred_delta_p - y_delta_p, dim=1)
            pos_errors.append(pos_error.cpu().numpy())
            
            # Compute quaternion angle error
            quat_angle = quaternion_angle_error(y_delta_q, pred_delta_q) * 180.0 / np.pi
            quat_angles.append(quat_angle.cpu().numpy())
    
    # Average metrics
    val_loss = np.mean(val_losses)
    val_pos_rmse = np.sqrt(np.mean(np.concatenate(pos_errors) ** 2))
    val_quat_angle = np.mean(np.concatenate(quat_angles))
    
    model.train()
    return val_loss, val_pos_rmse, val_quat_angle


def process_into_windows(gyro_data, acc_data, pos_data, ori_data, window_size=200, prediction_span=10, stride=10):
    """
    Process raw sensor and ground truth data into windows for training.
    
    Args:
        gyro_data: Gyroscope data (N, 3)
        acc_data: Accelerometer data (N, 3)
        pos_data: Position data (N, 3)
        ori_data: Orientation data as quaternions (N, 4)
        window_size: Size of each window for input data
        prediction_span: Number of samples for prediction (short-term prediction span)
        stride: Step size between windows
        
    Returns:
        x_gyro: Windowed gyroscope data
        x_acc: Windowed accelerometer data
        y_delta_p: Position change for the prediction span
        y_delta_q: Orientation change for the prediction span (as quaternion)
        init_p: Initial position for each window
        init_q: Initial orientation for each window
    """
    print("Processing data into windows with short-term prediction targets...")
    
    # Calculate maximum valid window index
    max_window_idx = len(gyro_data) - window_size + 1
    print(f"Max window index: {max_window_idx}")
    
    if max_window_idx <= 0:
        raise ValueError(f"Window size ({window_size}) too large for data length ({len(gyro_data)})")
    
    # Use sliding window to create data windows
    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []
    init_p = []
    init_q = []
    
    # Create windows
    for i in range(0, max_window_idx, stride):
        # Input windows (full context window)
        gyro_window = gyro_data[i:i+window_size]
        acc_window = acc_data[i:i+window_size]
        
        # Initial position and orientation (at center of window)
        center_idx = window_size // 2
        p_start = pos_data[i + center_idx - prediction_span//2]
        q_start = ori_data[i + center_idx - prediction_span//2]
        
        # End position and orientation (prediction_span samples later)
        p_end = pos_data[i + center_idx + prediction_span//2]
        q_end = ori_data[i + center_idx + prediction_span//2]
        
        # Calculate deltas for the prediction span (not the full window)
        delta_p = p_end - p_start
        delta_q = compute_delta_quaternion(q_start, q_end)
        
        # Append to lists
        x_gyro.append(gyro_window)
        x_acc.append(acc_window)
        y_delta_p.append(delta_p)
        y_delta_q.append(delta_q)
        init_p.append(p_start)
        init_q.append(q_start)
    
    # Convert to numpy arrays
    x_gyro = np.array(x_gyro)
    x_acc = np.array(x_acc)
    y_delta_p = np.array(y_delta_p)
    y_delta_q = np.array(y_delta_q)
    init_p = np.array(init_p)
    init_q = np.array(init_q)
    
    print(f"Created {len(x_gyro)} windows from data")
    print(f"Input shape: {x_gyro.shape}, Target shape: {y_delta_p.shape}")
    print(f"Position deltas predict motion over {prediction_span} samples")
    
    return x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q

def compute_delta_quaternion(q_start, q_end):
    """
    Compute the relative quaternion rotation from q_start to q_end.
    q_rel = q_end * q_start^(-1)
    
    Args:
        q_start: Starting quaternion [qw, qx, qy, qz]
        q_end: Ending quaternion [qw, qx, qy, qz]
        
    Returns:
        q_rel: Relative quaternion
    """
    # Extract components
    w1, x1, y1, z1 = q_start
    w2, x2, y2, z2 = q_end
    
    # Compute inverse of q_start (conjugate, since it's a unit quaternion)
    w1_inv, x1_inv, y1_inv, z1_inv = w1, -x1, -y1, -z1
    
    # Multiply q_end * q_start_inv
    w = w2*w1_inv - x2*x1_inv - y2*y1_inv - z2*z1_inv
    x = w2*x1_inv + x2*w1_inv + y2*z1_inv - z2*y1_inv
    y = w2*y1_inv - x2*z1_inv + y2*w1_inv + z2*x1_inv
    z = w2*z1_inv + x2*y1_inv - y2*x1_inv + z2*w1_inv
    
    return np.array([w, x, y, z])

def create_train_val_split(x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q, val_ratio=0.2, batch_size=32, shuffle_train=True):
    """
    Create train/validation split from a single dataset with proper shuffling.
    Modified to use all data for both training and validation for overfitting.
    """
    # Get total number of samples
    num_samples = len(x_gyro)
    
    # Ensure batch size is smaller than dataset size
    batch_size = min(batch_size, max(2, num_samples // 2))  # Ensure batch size is at least 2
    
    # Use all data for both training and validation
    train_dataset = TensorDataset(
        x_gyro, 
        x_acc, 
        y_delta_p, 
        y_delta_q,
        torch.FloatTensor(init_p),
        torch.FloatTensor(init_q)
    )
    
    # Use same dataset for validation
    val_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Using all {len(train_dataset)} samples for both training and validation")
    
    return train_loader, val_loader

def load_pretrained_model(model_path):
    """
    Load a pre-trained model and determine its type.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to determine model type
    model_type = checkpoint.get('model_type', None)
    
    # If not explicitly stored, try to infer from state dict
    if model_type is None and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if any('tcn' in key for key in state_dict.keys()):
            model_type = 'tcn'
        elif any('lstm' in key for key in state_dict.keys()):
            model_type = 'lstm'
        else:
            print("Warning: Could not determine model type from state dict. Defaulting to 'lstm'.")
            model_type = 'lstm'
    
    # Create model
    model = create_model(model_type)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try direct loading if no 'model_state_dict' key
        model.load_state_dict(checkpoint)
        
    print(f"Loaded {model_type} model successfully")
    
    # Move model to device
    model = model.to(device)
    
    return model, model_type


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one epoch of training."""
    model.train()
    total_loss = 0
    total_pos_rmse = 0
    total_quat_angle = 0
    batch_count = 0
    
    # Fix: Update to unpack all 6 values from dataloader
    for x_gyro, x_acc, y_delta_p, y_delta_q, _, _ in dataloader:
        # Move data to device
        x_gyro = x_gyro.to(device)
        x_acc = x_acc.to(device)
        y_delta_p = y_delta_p.to(device)
        y_delta_q = y_delta_q.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pos_pred, quat_pred = model(x_gyro, x_acc)
        
        # Calculate loss
        loss = criterion([pos_pred, quat_pred], [y_delta_p, y_delta_q])
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Position RMSE
            pos_rmse = torch.sqrt(torch.mean(torch.sum((pos_pred - y_delta_p) ** 2, dim=1)))
            
            # Quaternion angle error
            quat_error = quaternion_angle_error(y_delta_q, quat_pred)
            quat_angle = quat_error.mean() * 180.0 / np.pi  # Convert to degrees
        
        # Accumulate batch metrics
        batch_size = x_gyro.size(0)
        total_loss += loss.item() * batch_size
        total_pos_rmse += pos_rmse.item() * batch_size
        total_quat_angle += quat_angle.item() * batch_size
        batch_count += batch_size
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count
    avg_pos_rmse = total_pos_rmse / batch_count
    avg_quat_angle = total_quat_angle / batch_count
    
    return avg_loss, avg_pos_rmse, avg_quat_angle

def validate(model, dataloader, criterion, device):
    """
    Validate the model on validation data.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        
    Returns:
        val_loss: Average validation loss
        val_pos_rmse: Position RMSE
        val_quat_angle: Average quaternion angle error (degrees)
    """
    model.eval()
    val_losses = []
    pos_errors = []
    quat_angles = []
    
    with torch.no_grad():
        for x_gyro, x_acc, y_delta_p, y_delta_q, _, _ in dataloader:
            # Move data to device
            x_gyro = x_gyro.to(device)
            x_acc = x_acc.to(device)
            y_delta_p = y_delta_p.to(device)
            y_delta_q = y_delta_q.to(device)
            
            # Forward pass
            pred_delta_p, pred_delta_q = model(x_gyro, x_acc)
            
            # Compute loss
            y_true = [y_delta_p, y_delta_q]
            y_pred = [pred_delta_p, pred_delta_q]
            loss = criterion(y_pred, y_true)
            val_losses.append(loss.item())
            
            # Compute position error
            pos_error = torch.norm(pred_delta_p - y_delta_p, dim=1)
            pos_errors.append(pos_error.cpu().numpy())
            
            # Compute quaternion angle error
            quat_angle = quaternion_angle_error(y_delta_q, pred_delta_q) * 180.0 / np.pi
            quat_angles.append(quat_angle.cpu().numpy())
    
    # Average metrics
    val_loss = np.mean(val_losses)
    val_pos_rmse = np.sqrt(np.mean(np.concatenate(pos_errors) ** 2))
    val_quat_angle = np.mean(np.concatenate(quat_angles))
    
    model.train()
    return val_loss, val_pos_rmse, val_quat_angle


def train_model(model, train_loader, val_loader, learning_rate=1e-4, epochs=50, output_path=None):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        output_path: Path to save the best model
        
    Returns:
        model: Trained model
        train_history: Training metrics history
        val_history: Validation metrics history
    """
    # Initialize criterion and optimizer
    criterion = CustomMultiLossLayer(nb_outputs=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    
    train_history = {
        'loss': [], 'pos_rmse': [], 'quat_angle': [],
        'detailed_metrics': []  # Will store detailed metrics every 5 epochs
    }
    val_history = {
        'loss': [], 'pos_rmse': [], 'quat_angle': [],
        'detailed_metrics': []  # Will store detailed metrics every 5 epochs
    }
    
    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_pos_rmse, train_quat_angle = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_pos_rmse, val_quat_angle = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.6f}, RMSE: {train_pos_rmse:.4f}m, Quat: {train_quat_angle:.2f}° | "
              f"Val Loss: {val_loss:.6f}, RMSE: {val_pos_rmse:.4f}m, Quat: {val_quat_angle:.2f}°")
        
        # Record history
        train_history['loss'].append(train_loss)
        train_history['pos_rmse'].append(train_pos_rmse)
        train_history['quat_angle'].append(train_quat_angle)
        val_history['loss'].append(val_loss)
        val_history['pos_rmse'].append(val_pos_rmse)
        val_history['quat_angle'].append(val_quat_angle)
        
        # Calculate detailed metrics every 5 epochs - includes trajectory integration and drift
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"\nCalculating detailed metrics for epoch {epoch+1}...")
            
            # Calculate detailed metrics on validation set
            detailed_metrics = calculate_detailed_metrics(model, val_loader, device)
            
            # Store detailed metrics
            val_history['detailed_metrics'].append({
                'epoch': epoch + 1,
                **detailed_metrics
            })
            
            # Print detailed metrics
            print(f"Detailed Validation Metrics (Epoch {epoch+1}):")
            print(f"  Total Drift: {detailed_metrics['total_drift']:.4f}m")
            print(f"  Position RMSE: {detailed_metrics['rmse_position']:.4f}m")
            print(f"  Mean Position Error: {detailed_metrics['mean_position_error']:.4f}m")
            print(f"  Mean Quaternion Error: {detailed_metrics['mean_quaternion_error_deg']:.2f}°")
            print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model (val_loss: {val_loss:.6f})")
            
            # Save model if output path is provided
            if output_path:
                # Check if file exists and remove it first
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_type': model.__class__.__name__
                }, output_path + ".tmp")
                # Rename to avoid broken files if interrupted
                os.rename(output_path + ".tmp", output_path)
                print(f"Saved checkpoint to {output_path}")
                
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state based on validation loss")
    
    return model, train_history, val_history


def calculate_detailed_metrics(model, dataloader, device, prediction_span=10):
    """
    Calculate detailed metrics similar to those in evaluate_model.py.
    This version considers short-term predictions.
    """
    model.eval()
    
    # Collect all data for trajectory integration
    all_pred_delta_p = []
    all_pred_delta_q = []
    all_gt_delta_p = []
    all_gt_delta_q = []
    all_init_p = []
    all_init_q = []
    
    with torch.no_grad():
        for x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q in dataloader:
            # Move data to device
            x_gyro = x_gyro.to(device)
            x_acc = x_acc.to(device)
            
            # Forward pass
            pred_delta_p, pred_delta_q = model(x_gyro, x_acc)
            
            # Move predictions to CPU and convert to numpy
            all_pred_delta_p.append(pred_delta_p.cpu().numpy())
            all_pred_delta_q.append(pred_delta_q.cpu().numpy())
            all_gt_delta_p.append(y_delta_p.numpy())
            all_gt_delta_q.append(y_delta_q.numpy())
            all_init_p.append(init_p.numpy())
            all_init_q.append(init_q.numpy())
    
    # Concatenate all data
    pred_delta_p = np.vstack(all_pred_delta_p)
    pred_delta_q = np.vstack(all_pred_delta_q)
    gt_delta_p = np.vstack(all_gt_delta_p)
    gt_delta_q = np.vstack(all_gt_delta_q)
    init_p = np.vstack(all_init_p)
    init_q = np.vstack(all_init_q)
    
    # Directly compare delta predictions (no need for integration)
    delta_p_error = np.linalg.norm(pred_delta_p - gt_delta_p, axis=1)
    mean_delta_p_error = np.mean(delta_p_error)
    rmse_delta_p = np.sqrt(np.mean(np.square(delta_p_error)))
    
    # Quaternion angle errors
    q_errors_rad = []
    for i in range(len(pred_delta_q)):
        q_true = torch.tensor(gt_delta_q[i]).unsqueeze(0).float()
        q_pred = torch.tensor(pred_delta_q[i]).unsqueeze(0).float()
        angle_rad = quaternion_angle_error(q_true, q_pred).item()
        q_errors_rad.append(angle_rad)
    
    q_errors_deg = np.array(q_errors_rad) * 180.0 / np.pi
    mean_q_error_deg = np.mean(q_errors_deg)
    median_q_error_deg = np.median(q_errors_deg)
    
    # Still integrate full trajectory to assess overall drift
    # Assuming consecutive predictions with stride = prediction_span
    pred_trajectory = np.zeros((len(pred_delta_p), 3))
    gt_trajectory = np.zeros((len(gt_delta_p), 3))
    
    # Set initial position
    pred_trajectory[0] = init_p[0]
    gt_trajectory[0] = init_p[0]
    
    # Integrate trajectories (assuming stride equals prediction span)
    for i in range(1, len(pred_delta_p)):
        pred_trajectory[i] = pred_trajectory[i-1] + pred_delta_p[i-1]
        gt_trajectory[i] = gt_trajectory[i-1] + gt_delta_p[i-1]
    
    # Calculate position errors
    pos_error = np.linalg.norm(pred_trajectory - gt_trajectory, axis=1)
    mean_pos_error = np.mean(pos_error)
    rmse_pos = np.sqrt(np.mean(np.square(pos_error)))
    
    # Total drift (end-to-end error)
    total_drift = np.linalg.norm(pred_trajectory[-1] - gt_trajectory[-1])
    
    metrics = {
        'mean_position_error': mean_pos_error,
        'rmse_position': rmse_pos,
        'mean_delta_p_error': mean_delta_p_error,
        'rmse_delta_p': rmse_delta_p,
        'mean_quaternion_error_deg': mean_q_error_deg,
        'median_quaternion_error_deg': median_q_error_deg,
        'total_drift': total_drift
    }
    
    model.train()
    return metrics

def plot_learning_curves(train_history, val_history, output_prefix):
    """
    Plot learning curves for training and validation metrics.
    
    Args:
        train_history: Training metrics history
        val_history: Validation metrics history
        output_prefix: Prefix for output file paths
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot loss
    plt.figure(figsize=(12, 8))
    plt.plot(train_history['loss'], label='Training Loss')
    plt.plot(val_history['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot position RMSE
    plt.figure(figsize=(12, 8))
    plt.plot(train_history['pos_rmse'], label='Training Position RMSE')
    plt.plot(val_history['pos_rmse'], label='Validation Position RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (meters)')
    plt.title('Training and Validation Position RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_pos_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot quaternion angle error
    plt.figure(figsize=(12, 8))
    plt.plot(train_history['quat_angle'], label='Training Quaternion Error')
    plt.plot(val_history['quat_angle'], label='Validation Quaternion Error')
    plt.xlabel('Epoch')
    plt.ylabel('Angle Error (degrees)')
    plt.title('Training and Validation Quaternion Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_prefix}_quat_angle.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot detailed metrics if available
    if val_history['detailed_metrics']:
        epochs = [m['epoch'] for m in val_history['detailed_metrics']]
        
        # Plot total drift
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, [m['total_drift'] for m in val_history['detailed_metrics']], 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Total Drift (meters)')
        plt.title('Total Trajectory Drift over Training')
        plt.grid(True)
        plt.savefig(f'{output_prefix}_total_drift.png', dpi=300, bbox_inches='tight')
        plt.close()


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def initialize_new_model(model_type='lstm', enhanced=True):
    """
    Initialize a fresh model for training from scratch.
    
    Args:
        model_type: Type of model to create ('lstm' or 'tcn')
        enhanced: Whether to use enhanced model architecture
        
    Returns:
        model: Newly initialized model
    """
    print(f"Initializing new {model_type} model from scratch (enhanced={enhanced})")
    model = create_model(model_type=model_type, enhanced=enhanced)
    model = model.to(device)
    return model, model_type


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train or fine-tune inertial odometry model on custom dataset")
    parser.add_argument("--model_path", default=None, help="Path to pre-trained model checkpoint (if not training from scratch)")
    parser.add_argument("--output", default="imu_model", help="Base name for output files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--model_type", default="lstm", choices=["lstm", "tcn"], help="Model architecture to use")
    parser.add_argument("--from_scratch", action="store_true", help="Train model from scratch instead of fine-tuning")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced model architecture")
    parser.add_argument("--multi_sensor", action="store_true", help="Use multiple IMU sensors")
    parser.add_argument("--sensor_files", nargs='+', default=None, help="List of IMU sensor files for multi-sensor mode")
    parser.add_argument("--gt_file", default=None, help="Ground truth file for multi-sensor mode")
    parser.add_argument("--prediction_span", type=int, default=10, help="Number of samples to predict deltas for")
    args = parser.parse_args()
    
    # Define window size and prediction span
    PREDICTION_SPAN = args.prediction_span
    
    # Get timestamp for output files
    timestamp = get_timestamp()
    
    # Create output directory if it doesn't exist
    output_dir = Path("model_outputs")
    output_dir.mkdir(exist_ok=True)
    
    if args.multi_sensor:
        # Load multi-sensor dataset
        all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q, normalizers = load_multi_sensor_dataset(
            args.sensor_files, args.gt_file,
            window_size=WINDOW_SIZE, prediction_span=PREDICTION_SPAN, stride=STRIDE, normalize=True
        )
        
        # Create train/val split
        train_loader, val_loader = create_multi_sensor_train_val_split(
            all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q,
            val_ratio=args.val_ratio, batch_size=args.batch_size
        )
        
        # Initialize multi-sensor model
        model = create_model(
            model_type=args.model_type,
            enhanced=args.enhanced,
            multi_sensor=True,
            num_sensors=len(args.sensor_files)
        )
        model = model.to(device)
        
    else:
        # Load standard dataset
        x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q, normalizer = load_finetune_dataset(
            window_size=WINDOW_SIZE, prediction_span=PREDICTION_SPAN, stride=STRIDE, normalize=True
        )
        
        # Create train/val split
        train_loader, val_loader = create_train_val_split(
            x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q,
            val_ratio=args.val_ratio, batch_size=args.batch_size
        )
        
        # Initialize model - either from scratch or load pre-trained
        if args.from_scratch or args.model_path is None:
            print("Training new model from scratch")
            model, model_type = initialize_new_model(args.model_type, enhanced=args.enhanced)
        else:
            print(f"Fine-tuning pre-trained model from {args.model_path}")
            model, model_type = load_pretrained_model(args.model_path, enhanced=args.enhanced)
    
    
    # Prepare output paths
    prefix = "multi_" if args.multi_sensor else ("scratch_" if args.from_scratch else "finetune_")
    model_output = output_dir / f"{prefix}{args.output}_{timestamp}.pt"
    
    if args.multi_sensor:
        # Save all normalizers
        normalizer_output = output_dir / f"{prefix}{args.output}_{timestamp}_normalizers.pkl"
        with open(normalizer_output, 'wb') as f:
            pickle.dump(normalizers, f)
        print(f"Saved normalizers to {normalizer_output}")
    else:
        # Save single normalizer
        normalizer_output = output_dir / f"{prefix}{args.output}_{timestamp}_normalizer.pkl"
        if normalizer:
            normalizer.save(str(normalizer_output))
            print(f"Saved normalizer to {normalizer_output}")
    
    curve_output = output_dir / f"{prefix}{args.output}_{timestamp}"
    
    # Initialize criterion and optimizer
    criterion = CustomMultiLossLayer(nb_outputs=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Initialize training history
    train_history = {
        'loss': [], 'pos_rmse': [], 'quat_angle': [],
        'detailed_metrics': []
    }
    val_history = {
        'loss': [], 'pos_rmse': [], 'quat_angle': [],
        'detailed_metrics': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch - use appropriate function based on model type
        if args.multi_sensor:
            train_loss, train_pos_rmse, train_quat_angle = train_epoch_multi_sensor(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_pos_rmse, val_quat_angle = validate_multi_sensor(
                model, val_loader, criterion, device
            )
        else:
            train_loss, train_pos_rmse, train_quat_angle = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_pos_rmse, val_quat_angle = validate(
                model, val_loader, criterion, device
            )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.6f}, RMSE: {train_pos_rmse:.4f}m, Quat: {train_quat_angle:.2f}° | "
              f"Val Loss: {val_loss:.6f}, RMSE: {val_pos_rmse:.4f}m, Quat: {val_quat_angle:.2f}°")
        
        # Record history
        train_history['loss'].append(train_loss)
        train_history['pos_rmse'].append(train_pos_rmse)
        train_history['quat_angle'].append(train_quat_angle)
        val_history['loss'].append(val_loss)
        val_history['pos_rmse'].append(val_pos_rmse)
        val_history['quat_angle'].append(val_quat_angle)
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'multi_sensor': args.multi_sensor,
            'num_sensors': len(args.sensor_files) if args.multi_sensor else 1,
            'model_type': args.model_type
        }, str(model_output))
    
    # Plot learning curves
    plot_learning_curves(train_history, val_history, str(curve_output))
    
    print("Training complete!")
    
    
if __name__ == "__main__":
    main()
    
"""
python finetune_train.py lstm_model_with_norm_20250331_213713.pt --output mems_model --epochs 50 --batch_size 64 --lr 5e-5

python finetune_train.py --from_scratch --model_type lstm --output lstm_model_scratch --epochs 100 --batch_size 64 --lr 1e-3 --enhanced
"""