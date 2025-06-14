"""
inference.py
-----------
Real-time inference script for inertial odometry model.
Usage: python inference.py <model_checkpoint> <imu_data.csv> [--output <output.csv>] [--visualize]
"""

import argparse
import numpy as np
import pandas as pd
import torch
import quaternion
from torch.utils.data import Dataset, DataLoader
from training.initial_training.model import create_model
from data_preparation import quaternion_angle_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class IMUDataset(Dataset):
    """Dataset class for streaming IMU data"""
    def __init__(self, imu_data, window_size=200, stride=10):
        """
        Args:
            imu_data: numpy array of shape (N, 6) containing [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
            window_size: number of samples per window
            stride: step between consecutive windows
        """
        self.gyro_data = imu_data[:, :3]
        self.acc_data = imu_data[:, 3:]
        self.window_size = window_size
        self.stride = stride
        self.length = (imu_data.shape[0] - window_size) // stride + 1
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        gyro_window = self.gyro_data[start:end]
        acc_window = self.acc_data[start:end]
        return torch.FloatTensor(gyro_window), torch.FloatTensor(acc_window)

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get('model_type', 'lstm')  # Default to LSTM if not specified
    model = create_model(model_type=model_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def process_imu_file(imu_file):
    """Load and preprocess IMU data from CSV"""
    df = pd.read_csv(imu_file)
    
    # Assuming CSV has columns: timestamp, gx, gy, gz, ax, ay, az
    # Adjust as needed for your IMU format
    imu_data = df[['gx', 'gy', 'gz', 'ax', 'ay', 'az']].values
    
    # Optionally: Add data normalization here using saved normalization parameters
    return imu_data

def run_inference(model, dataloader, device, init_pose=None):
    """
    Run inference on IMU data
    Args:
        model: Trained model
        dataloader: DataLoader providing IMU windows
        device: 'cuda' or 'cpu'
        init_pose: Optional initial position and orientation (position, quaternion)
                   If None, will assume start at origin with identity orientation
    
    Returns:
        poses: List of (position, quaternion) tuples
        timestamps: List of timestamps for each pose (if available)
    """
    poses = []
    
    with torch.no_grad():
        for gyro, acc in dataloader:
            gyro, acc = gyro.to(device), acc.to(device)
            
            # Add batch dimension if not present
            if len(gyro.shape) == 2:
                gyro = gyro.unsqueeze(0)
                acc = acc.unsqueeze(0)
                
            # Run model inference
            delta_p, delta_q = model(gyro, acc)
            
            # Convert to numpy
            delta_p = delta_p.cpu().numpy()
            delta_q = delta_q.cpu().numpy()
            
            # For first window, use initial pose if provided
            if len(poses) == 0:
                if init_pose is not None:
                    current_p, current_q = init_pose
                else:
                    current_p = np.zeros(3)
                    current_q = np.array([1, 0, 0, 0])  # Identity quaternion
            else:
                # Update position and orientation by integrating deltas
                last_p, last_q = poses[-1]
                
                # Convert quaternions to numpy quaternion objects for easy math
                q_last = quaternion.from_float_array(last_q)
                q_delta = quaternion.from_float_array(delta_q[0])
                
                # Update orientation: q_new = q_last * q_delta
                current_q = quaternion.as_float_array(q_last * q_delta)
                
                # Update position: p_new = p_last + R(q_last) * delta_p
                R = quaternion.as_rotation_matrix(q_last)
                current_p = last_p + R @ delta_p[0]
            
            poses.append((current_p, current_q))
    
    return poses

def save_to_csv(poses, output_file, timestamps=None):
    """Save poses to CSV file"""
    positions = [p for p, q in poses]
    quaternions = [q for p, q in poses]
    
    df = pd.DataFrame({
        'timestamp': timestamps if timestamps is not None else np.arange(len(poses)),
        'pos_x': [p[0] for p in positions],
        'pos_y': [p[1] for p in positions],
        'pos_z': [p[2] for p in positions],
        'quat_w': [q[0] for q in quaternions],
        'quat_x': [q[1] for q in quaternions],
        'quat_y': [q[2] for q in quaternions],
        'quat_z': [q[3] for q in quaternions]
    })
    
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

def visualize_trajectory(poses):
    """3D plot of the estimated trajectory"""
    positions = np.array([p for p, q in poses])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            label='Estimated Trajectory')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title('Inertial Odometry Trajectory')
    ax.legend()
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained inertial odometry model')
    parser.add_argument('checkpoint', help='Path to model checkpoint file')
    parser.add_argument('imu_data', help='Path to IMU data CSV file')
    parser.add_argument('--output', default='output_poses.csv', help='Output CSV file path')
    parser.add_argument('--visualize', action='store_true', help='Show 3D trajectory visualization')
    parser.add_argument('--window_size', type=int, default=200, help='Window size for inference')
    parser.add_argument('--stride', type=int, default=10, help='Stride between windows')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)

    # Load and prepare IMU data
    print("Loading IMU data...")
    imu_data = process_imu_file(args.imu_data)
    
    # Create dataset and dataloader
    dataset = IMUDataset(imu_data, window_size=args.window_size, stride=args.stride)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference
    print("Running inference...")
    poses = run_inference(model, dataloader, device)

    # Save results
    save_to_csv(poses, args.output)

    # Visualize if requested
    if args.visualize:
        visualize_trajectory(poses)

if __name__ == '__main__':
    main()