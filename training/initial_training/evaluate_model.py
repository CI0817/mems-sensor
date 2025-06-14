"""
evaluate_model.py
----------------
This script loads a trained model from a checkpoint, runs inference on IMU data,
and compares the predicted trajectory with ground truth.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation

# Import from project modules
from model import create_model
from data_preparation import quaternion_angle_error
# Fix the import to use the correct function name
from training.finetune_training.finetune_train import process_into_windows, compute_delta_quaternion

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
WINDOW_SIZE = 200
MODEL_PATH = 'model_outputs/scratch_lstm_model_scratch_bigger_20250420_115122.pt'  # Path to the trained model
DATA_DIR = 'finetune_dataset'
OUTPUT_DIR = 'evaluation_results_nonoverlap_bigModel'
STRIDE = 10  # Using overlapping windows for continuous trajectory
PREDICTION_SPAN = 10  # Number of samples for prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained inertial odometry model')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                    help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                    help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                    help='Directory to save evaluation results')
    parser.add_argument('--stride', type=int, default=STRIDE,
                    help='Stride for window creation (default: 10)')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE,
                    help='Window size for input data (default: 200)')
    parser.add_argument('--prediction_span', type=int, default=PREDICTION_SPAN,
                    help='Number of samples for prediction (default: 10)')
    parser.add_argument('--num_sensors', type=int, default=4,
                    help='Number of IMU sensors (for multi-sensor models)')
    parser.add_argument('--sensor_files', nargs='+', default=None,
                    help='List of IMU sensor files (relative to data_dir)')
    parser.add_argument('--gt_file', type=str, default=None,
                    help='Ground truth file (relative to data_dir)')
    return parser.parse_args()

def ensure_dir_exists(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# def load_model(model_path):
#     """Load the trained model from checkpoint."""
#     print(f"Loading model from {model_path}")
    
#     # Create model
#     model = create_model()
    
#     # Load checkpoint
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Model loaded from epoch {checkpoint['epoch']}")
#     else:
#         raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
#     model.to(device)
#     model.eval()  # Set to evaluation mode
    
#     return model

def load_model(model_path):
    """Load the trained model and normalizer from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint first to check model type
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if this is a multi-sensor model
        is_multi_sensor = checkpoint.get('multi_sensor', False)
        num_sensors = checkpoint.get('num_sensors', 4)
        model_type = checkpoint.get('model_type', 'lstm')
        is_enhanced = model_type == 'EnhancedLSTMModel' or ('_enhanced' in model_path) or ('_bigger' in model_path)
        
        # Create the appropriate model
        model = create_model(
            model_type='lstm', 
            enhanced=is_enhanced,
            multi_sensor=is_multi_sensor,
            num_sensors=num_sensors
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Model loaded from epoch {epoch} (multi_sensor={is_multi_sensor}, num_sensors={num_sensors})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded (multi_sensor={is_multi_sensor}, num_sensors={num_sensors})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    # Also load normalizer(s)
    normalizer_path = model_path.replace('.pt', '_normalizer.pkl')
    multi_normalizer_path = model_path.replace('.pt', '_normalizers.pkl')
    
    normalizers = None
    if os.path.exists(normalizer_path):
        from data_preparation import IMUDataNormalizer
        try:
            normalizer = IMUDataNormalizer()
            normalizer.load(normalizer_path)
            normalizers = normalizer
            print(f"Loaded normalizer from {normalizer_path}")
        except Exception as e:
            print(f"Error loading normalizer: {e}")
            with open(normalizer_path, 'rb') as f:
                normalizers = pickle.load(f)
            print(f"Loaded normalizer using pickle from {normalizer_path}")
    elif os.path.exists(multi_normalizer_path):
        import pickle
        with open(multi_normalizer_path, 'rb') as f:
            normalizers = pickle.load(f)
        print(f"Loaded multi-sensor normalizers from {multi_normalizer_path}")
    else:
        print("Warning: Could not find normalizer file. Inference may be incorrect without normalization.")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, is_multi_sensor, normalizers

def load_data(data_dir, num_sensors=4):
    """Load multiple IMU sensor data and ground truth."""
    sensor_files = []
    all_gyro_data = []
    all_acc_data = []
    
    # Load all IMU sensor files
    for i in range(1, num_sensors + 1):
        imu_file = os.path.join(data_dir, f'MEMS_4_log_2_imu_{i}_processed.csv')
        sensor_files.append(imu_file)
        
        # Load data
        imu_data = pd.read_csv(imu_file)
        
        # Extract columns
        gyro_data = imu_data[[f'gyro_{i}_x', f'gyro_{i}_y', f'gyro_{i}_z']].values
        acc_data = imu_data[[f'acc_{i}_x', f'acc_{i}_y', f'acc_{i}_z']].values
        
        all_gyro_data.append(gyro_data)
        all_acc_data.append(acc_data)
    
    # Load ground truth data
    gt_file = os.path.join(data_dir, 'EKF_processed_interpolated.csv')
    gt_data = pd.read_csv(gt_file)
    pos_data = gt_data[['Displacement X (m)', 'Displacement Y (m)', 'Displacement Z (m)']].values
    ori_data = gt_data[['q0(w)', 'q1(x)', 'q2(y)', 'q3(z)']].values
    
    print(f"Data loaded - IMU samples: {len(all_gyro_data[0])}, GT samples: {len(pos_data)}")
    
    return all_gyro_data, all_acc_data, pos_data, ori_data

def process_data(all_gyro_data, all_acc_data, pos_data, ori_data, window_size, prediction_span, stride):
    """Process multi-sensor data into windows for model input with short-term prediction targets."""
    all_x_gyro = []
    all_x_acc = []
    
    for i in range(len(all_gyro_data)):
        # Process each sensor's data into windows
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
    
    print(f"Processed {len(y_delta_p)} windows with stride {stride}")
    return all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q

def run_inference(model, all_x_gyro, all_x_acc):
    """Run inference with the multi-sensor model."""
    batch_size = 32
    n_samples = len(all_x_gyro[0])
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    pred_delta_p_list = []
    pred_delta_q_list = []
    
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Get batch for each sensor and move to device
            gyro_batch_list = []
            acc_batch_list = []
            
            for j in range(len(all_x_gyro)):
                gyro_batch = torch.tensor(all_x_gyro[j][start_idx:end_idx]).float().to(device)
                acc_batch = torch.tensor(all_x_acc[j][start_idx:end_idx]).float().to(device)
                
                gyro_batch_list.append(gyro_batch)
                acc_batch_list.append(acc_batch)
            
            # Forward pass
            pred_delta_p, pred_delta_q = model(gyro_batch_list, acc_batch_list)
            
            # Move predictions to CPU and convert to numpy
            pred_delta_p_list.append(pred_delta_p.cpu().numpy())
            pred_delta_q_list.append(pred_delta_q.cpu().numpy())
    
    # Concatenate batch predictions
    pred_delta_p = np.vstack(pred_delta_p_list)
    pred_delta_q = np.vstack(pred_delta_q_list)
    
    return pred_delta_p, pred_delta_q

def integrate_trajectory_short_term(init_p, pred_delta_p, stride=10, prediction_span=10):
    """
    Integrate predicted position deltas to form a continuous trajectory.
    This version matches the integration in finetune_train.py's calculate_detailed_metrics.
    
    Args:
        init_p: Initial positions for each window
        pred_delta_p: Predicted position deltas for each window
        stride: Window stride for processing (default: 10)
        prediction_span: Number of samples for prediction (default: 10)
    
    Returns:
        trajectory: Integrated trajectory
    """
    n_windows = len(pred_delta_p)
    trajectory = np.zeros((n_windows, 3))
    
    # Set initial position
    trajectory[0] = init_p[0]
    print(f"Integration - Initial position: {trajectory[0]}")
    
    # Integrate trajectory (matching finetune_train.py's approach)
    for i in range(1, n_windows):
        trajectory[i] = trajectory[i-1] + pred_delta_p[i-1]
        
        # Periodically print progress
        if i % 100 == 0 or i == n_windows-1:
            print(f"Window {i}/{n_windows-1}, Position: {trajectory[i]}")
    
    return trajectory

def integrate_trajectory(init_p, pred_delta_p):
    """
    Integrate predicted deltas to form a full trajectory.
    
    Parameters:
        init_p: Initial positions array of shape (n_samples, 3)
        pred_delta_p: Predicted position deltas array of shape (n_samples, 3)
        
    Returns:
        trajectory: Integrated trajectory of shape (n_samples, 3)
    """
    n_samples = len(pred_delta_p)
    trajectory = np.zeros((n_samples, 3))
    
    # Set initial position
    trajectory[0] = init_p[0]
    
    # Debug info
    print(f"Integration - Initial position: {trajectory[0]}")
    print(f"First few deltas: {pred_delta_p[:3]}")
    
    # Integrate trajectory - accumulate deltas from each step
    for i in range(1, n_samples):
        trajectory[i] = trajectory[i-1] + pred_delta_p[i-1]
        # Periodically print progress
        if i % 10 == 0 or i == n_samples-1:
            print(f"Position at step {i}/{n_samples-1}: {trajectory[i]}")
    
    return trajectory

def calculate_errors(pred_trajectory, gt_trajectory, pred_delta_p, gt_delta_p, pred_delta_q, gt_delta_q):
    """Calculate error metrics."""
    # Position trajectory error
    pos_error = np.linalg.norm(pred_trajectory - gt_trajectory, axis=1)
    mean_pos_error = np.mean(pos_error)
    median_pos_error = np.median(pos_error)
    rmse_pos = np.sqrt(np.mean(np.square(pos_error)))
    
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(gt_delta_p):
        gt_delta_p = gt_delta_p.cpu().numpy()
    if torch.is_tensor(gt_delta_q):
        gt_delta_q = gt_delta_q.cpu().numpy()
    
    # Delta position error
    delta_p_error = np.linalg.norm(pred_delta_p - gt_delta_p, axis=1)
    mean_delta_p_error = np.mean(delta_p_error)
    rmse_delta_p = np.sqrt(np.mean(np.square(delta_p_error)))
    
    # Convert quaternion errors to angles (degrees)
    q_errors_rad = []
    for i in range(len(pred_delta_q)):
        q_true = torch.tensor(gt_delta_q[i]).unsqueeze(0).float()
        q_pred = torch.tensor(pred_delta_q[i]).unsqueeze(0).float()
        angle_rad = quaternion_angle_error(q_true, q_pred).item()
        q_errors_rad.append(angle_rad)
    
    q_errors_deg = np.array(q_errors_rad) * 180.0 / np.pi
    mean_q_error_deg = np.mean(q_errors_deg)
    median_q_error_deg = np.median(q_errors_deg)
    
    errors = {
        'mean_position_error': mean_pos_error,
        'median_position_error': median_pos_error,
        'rmse_position': rmse_pos,
        'mean_delta_p_error': mean_delta_p_error,
        'rmse_delta_p': rmse_delta_p,
        'mean_quaternion_error_deg': mean_q_error_deg,
        'median_quaternion_error_deg': median_q_error_deg,
        'total_drift': np.linalg.norm(pred_trajectory[-1] - gt_trajectory[-1])
    }
    
    return errors, pos_error, q_errors_deg

def plot_trajectory(pred_trajectory, gt_trajectory, output_dir):
    """Plot 3D trajectory comparison."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            label='Predicted', color='blue', linewidth=2)
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
            label='Ground Truth', color='red', linewidth=2)
    
    # Mark start and end points
    ax.scatter(pred_trajectory[0, 0], pred_trajectory[0, 1], pred_trajectory[0, 2], 
               color='green', s=100, marker='o', label='Start')
    ax.scatter(pred_trajectory[-1, 0], pred_trajectory[-1, 1], pred_trajectory[-1, 2], 
               color='purple', s=100, marker='x', label='Pred End')
    ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
               color='black', s=100, marker='x', label='GT End')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Trajectory Comparison (Non-overlapping Windows)')
    ax.legend()
    
    # Adjust view for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, '3d_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2D projections
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # XY projection
    axes[0].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], label='Predicted', color='blue')
    axes[0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground Truth', color='red')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Y [m]')
    axes[0].set_title('XY Projection')
    axes[0].legend()
    axes[0].grid(True)
    
    # XZ projection
    axes[1].plot(pred_trajectory[:, 0], pred_trajectory[:, 2], label='Predicted', color='blue')
    axes[1].plot(gt_trajectory[:, 0], gt_trajectory[:, 2], label='Ground Truth', color='red')
    axes[1].set_xlabel('X [m]')
    axes[1].set_ylabel('Z [m]')
    axes[1].set_title('XZ Projection')
    axes[1].legend()
    axes[1].grid(True)
    
    # YZ projection
    axes[2].plot(pred_trajectory[:, 1], pred_trajectory[:, 2], label='Predicted', color='blue')
    axes[2].plot(gt_trajectory[:, 1], gt_trajectory[:, 2], label='Ground Truth', color='red')
    axes[2].set_xlabel('Y [m]')
    axes[2].set_ylabel('Z [m]')
    axes[2].set_title('YZ Projection')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_projections.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_errors(pos_error, q_errors_deg, output_dir):
    """Plot error distributions."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Position error histogram
    axes[0].hist(pos_error, bins=30, alpha=0.7)
    axes[0].axvline(np.mean(pos_error), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(pos_error):.3f} m')
    axes[0].axvline(np.median(pos_error), color='green', linestyle='--', 
                   label=f'Median: {np.median(pos_error):.3f} m')
    axes[0].set_xlabel('Position Error [m]')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Position Error Distribution')
    axes[0].legend()
    axes[0].grid(True)
    
    # Quaternion error histogram
    axes[1].hist(q_errors_deg, bins=30, alpha=0.7)
    axes[1].axvline(np.mean(q_errors_deg), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(q_errors_deg):.3f} deg')
    axes[1].axvline(np.median(q_errors_deg), color='green', linestyle='--', 
                   label=f'Median: {np.median(q_errors_deg):.3f} deg')
    axes[1].set_xlabel('Quaternion Angle Error [degrees]')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Quaternion Error Distribution')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_error(pred_trajectory, gt_trajectory, output_dir):
    """Plot cumulative position error over time."""
    distances = np.zeros(len(pred_trajectory))
    for i in range(len(pred_trajectory)):
        distances[i] = np.linalg.norm(pred_trajectory[i] - gt_trajectory[i])
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.xlabel('Window Index')
    plt.ylabel('Cumulative Position Error [m]')
    plt.title('Cumulative Position Error Over Time')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'cumulative_error.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(errors, output_dir):
    """Save error metrics to file."""
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write('Evaluation Metrics (Non-overlapping Windows)\n')
        f.write('====================================\n\n')
        for key, value in errors.items():
            f.write(f"{key}: {value:.6f}\n")


def build_gt_trajectory(pos_data, window_size, prediction_span, stride, n_windows):
    """Build ground truth trajectory in the same way as integrated predicted trajectory."""
    trajectory = np.zeros((n_windows, 3))
    
    # Start with the same initial position as the predicted trajectory
    trajectory[0] = pos_data[0]  
    
    # Use the deltas between consecutive ground truth points
    for i in range(1, min(n_windows, len(pos_data))):
        if i * stride < len(pos_data):
            # Use actual position changes from ground truth
            center_idx = window_size // 2
            prev_idx = max(0, i-1) * stride + center_idx - prediction_span//2
            curr_idx = i * stride + center_idx - prediction_span//2
            
            if curr_idx < len(pos_data) and prev_idx < len(pos_data):
                # Use the same delta calculation as in process_into_windows
                delta_p = pos_data[curr_idx] - pos_data[prev_idx]
                trajectory[i] = trajectory[i-1] + delta_p
    
    return trajectory

def main():
    args = parse_args()
    ensure_dir_exists(args.output_dir)
    
    # Load model
    model, is_multi_sensor, normalizers = load_model(args.model)
    
    # Load appropriate data based on model type
    if is_multi_sensor:
        # Load multi-sensor data
        print("Loading multi-sensor data for evaluation...")
        if args.sensor_files and args.gt_file:
            # Use provided file paths
            print(f"Using provided sensor files: {args.sensor_files}")
            print(f"Using provided GT file: {args.gt_file}")
            
            all_gyro_data = []
            all_acc_data = []
            
            # Load each sensor file
            for i, sensor_file in enumerate(args.sensor_files):
                full_path = os.path.join(args.data_dir, sensor_file)
                print(f"Loading sensor {i+1} from {full_path}")
                imu_data = pd.read_csv(full_path)
                
                # Extract gyroscope and accelerometer data
                gyro_data = imu_data[[f'gyro_{i+1}_x', f'gyro_{i+1}_y', f'gyro_{i+1}_z']].values
                acc_data = imu_data[[f'acc_{i+1}_x', f'acc_{i+1}_y', f'acc_{i+1}_z']].values
                
                all_gyro_data.append(gyro_data)
                all_acc_data.append(acc_data)
                
            # Load ground truth data
            gt_path = os.path.join(args.data_dir, args.gt_file)
            print(f"Loading ground truth from {gt_path}")
            gt_data = pd.read_csv(gt_path)
            pos_data = gt_data[['Displacement X (m)', 'Displacement Y (m)', 'Displacement Z (m)']].values
            ori_data = gt_data[['q0(w)', 'q1(x)', 'q2(y)', 'q3(z)']].values
        else:
            # Use default file patterns
            all_gyro_data, all_acc_data, pos_data, ori_data = load_data(args.data_dir, args.num_sensors)
        
        # Process data with the specified parameters
        all_x_gyro, all_x_acc, y_delta_p, y_delta_q, init_p, init_q = process_data(
            all_gyro_data, all_acc_data, pos_data, ori_data,
            window_size=args.window_size,
            prediction_span=args.prediction_span,
            stride=args.stride
        )
        
        # Run inference
        if normalizers is not None and isinstance(normalizers, list):
            print("Applying multi-sensor normalization")
            for i in range(min(len(all_x_gyro), len(normalizers))):
                all_x_gyro[i], all_x_acc[i] = normalizers[i].transform(all_x_gyro[i], all_x_acc[i])
                print(f"Normalized sensor {i+1} data")
        
        # Run inference with normalized data
        pred_delta_p, pred_delta_q = run_inference(model, all_x_gyro, all_x_acc)
    else:
        # Load single-sensor data (use the first IMU file)
        imu_file = os.path.join(args.data_dir, 'MEMS_4_log_2_imu_1_processed.csv')
        gt_file = os.path.join(args.data_dir, 'EKF_processed_interpolated.csv')
        
        print(f"Loading IMU data from {imu_file}")
        imu_data = pd.read_csv(imu_file)
        
        print(f"Loading ground truth data from {gt_file}")
        gt_data = pd.read_csv(gt_file)
        
        # Extract relevant columns
        gyro_data = imu_data[['gyro_1_x', 'gyro_1_y', 'gyro_1_z']].values
        acc_data = imu_data[['acc_1_x', 'acc_1_y', 'acc_1_z']].values
        pos_data = gt_data[['Displacement X (m)', 'Displacement Y (m)', 'Displacement Z (m)']].values
        ori_data = gt_data[['q0(w)', 'q1(x)', 'q2(y)', 'q3(z)']].values
        
        # Process data with the specified parameters
        x_gyro, x_acc, y_delta_p, y_delta_q, init_p, init_q = process_into_windows(
            gyro_data, acc_data, pos_data, ori_data,
            window_size=args.window_size,
            prediction_span=args.prediction_span,
            stride=args.stride
        )
        
        # Run inference with single sensor model
        batch_size = 32
        n_samples = len(x_gyro)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        pred_delta_p_list = []
        pred_delta_q_list = []
        
        with torch.no_grad():
            for i in tqdm(range(n_batches)):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # Get batch and move to device
                gyro_batch = torch.tensor(x_gyro[start_idx:end_idx]).float().to(device)
                acc_batch = torch.tensor(x_acc[start_idx:end_idx]).float().to(device)
                
                # Forward pass
                pred_delta_p_batch, pred_delta_q_batch = model(gyro_batch, acc_batch)
                
                # Move predictions to CPU and convert to numpy
                pred_delta_p_list.append(pred_delta_p_batch.cpu().numpy())
                pred_delta_q_list.append(pred_delta_q_batch.cpu().numpy())
        
        # Concatenate batch predictions
        pred_delta_p = np.vstack(pred_delta_p_list)
        pred_delta_q = np.vstack(pred_delta_q_list)
    
    # Build trajectories using the short-term integration method
    n_windows = len(pred_delta_p)
    pred_trajectory = integrate_trajectory_short_term(
        init_p, pred_delta_p, args.stride, args.prediction_span
    )
    
    # Build ground truth trajectory
    gt_trajectory = build_gt_trajectory(
        pos_data, args.window_size, args.prediction_span, args.stride, n_windows
    )
    
    # Trim trajectories to the same length (shorter of the two)
    min_length = min(len(pred_trajectory), len(gt_trajectory))
    pred_trajectory = pred_trajectory[:min_length]
    gt_trajectory = gt_trajectory[:min_length]
    
    print(f"Final trajectory length: {len(pred_trajectory)}")
    
    # Calculate errors
    errors, pos_error, q_errors_deg = calculate_errors(
        pred_trajectory, gt_trajectory, pred_delta_p, y_delta_p, pred_delta_q, y_delta_q
    )
    
    # Plot results
    plot_trajectory(pred_trajectory, gt_trajectory, args.output_dir)
    plot_errors(pos_error, q_errors_deg, args.output_dir)
    plot_cumulative_error(pred_trajectory, gt_trajectory, args.output_dir)
    
    # Save results
    save_results(errors, args.output_dir)
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")
    print("\nSummary of results:")
    print(f"Position RMSE: {errors['rmse_position']:.4f} m")
    print(f"Mean position error: {errors['mean_position_error']:.4f} m")
    print(f"Mean quaternion error: {errors['mean_quaternion_error_deg']:.4f} degrees")
    print(f"Total drift at end: {errors['total_drift']:.4f} m")

if __name__ == "__main__":
    main()
    
"""
python evaluate_model.py --model model_outputs/multi_multi_sensor_model_10_20250421_150751.pt --output_dir evaluation_results_multi_sensor --num_sensors 4 --sensor_files MEMS_4_log_2_imu_1_processed.csv MEMS_4_log_2_imu_2_processed.csv MEMS_4_log_2_imu_3_processed.csv MEMS_4_log_2_imu_4_processed.csv --gt_file EKF_processed_interpolated.csv

"""