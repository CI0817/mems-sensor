"""
dataset.py
----------
This module provides:
  - Utility functions for sliding-window generation, interpolation, and quaternion/coordinate operations.
  - A PyTorch Dataset class for IMU data.
  - Functions to load various datasets (EuRoC MAV, OXIOD) and to build windowed sequences
    in different representations (6D with quaternions, 6D with Rodrigues vectors, 3D, and 2D).
"""

import numpy as np                         # Import NumPy for numerical operations
import pandas as pd                        # Import pandas for data manipulation and CSV reading
import quaternion                          # Import quaternion package for quaternion arithmetic
import scipy.interpolate                   # Import SciPy interpolation functions
from typing import List, Tuple             # Import type hints for function signatures

# ----------------------------
# Helper Functions & Utilities
# ----------------------------

def sliding_window_indices(total_length: int, window_size: int, stride: int) -> range:
    """Generate starting indices for sliding windows."""
    # Return a range of starting indices that allow creation of sliding windows
    return range(0, total_length - window_size - 1, stride)

def stack_windows(windows: List[np.ndarray]) -> np.ndarray:
    """Stack a list of windows into a numpy array."""
    # Convert the list of window arrays into a single NumPy array for batch processing
    return np.array(windows)

def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    # If the angle is less than -pi, add 2*pi to bring it within range
    if angle < -np.pi:
        return angle + 2 * np.pi
    # If the angle is greater than pi, subtract 2*pi to bring it within range
    elif angle > np.pi:
        return angle - 2 * np.pi
    # Otherwise, return the angle as is
    return angle

# ----------------------------
# Interpolation and Data Loading Functions
# ----------------------------

def interpolate_3dvector_linear(data: np.ndarray, input_ts: np.ndarray, 
                                output_ts: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a 3D vector (e.g., gyro or acc data) from the input timestamps
    to the output timestamps.
    
    Args:
        data: Data array of shape (N, 3) representing 3D vector measurements.
        input_ts: Array of input timestamps corresponding to the data.
        output_ts: Array of desired output timestamps to interpolate to.
        
    Returns:
        Interpolated data array evaluated at output_ts.
    """
    # Ensure that the number of data samples matches the number of input timestamps
    assert data.shape[0] == input_ts.shape[0], "Data and timestamps must have the same length."
    # Create an interpolation function along the time axis (axis=0)
    interp_func = scipy.interpolate.interp1d(input_ts, data, axis=0)
    # Interpolate the data at the specified output timestamps
    return interp_func(output_ts)

def load_euroc_mav_dataset(imu_file: str, gt_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and interpolate the EuRoC MAV dataset.
    
    Args:
        imu_file: Path to the IMU CSV file containing sensor measurements.
        gt_file: Path to the ground-truth CSV file containing position and orientation data.
        
    Returns:
        Tuple containing:
         - gyro_data: Interpolated gyroscope measurements.
         - acc_data: Interpolated accelerometer measurements.
         - pos_data: Position data from ground-truth.
         - ori_data: Orientation data (quaternions) from ground-truth.
    """
    # Read the ground-truth CSV file; expected format: timestamp, x, y, z, w, x, y, z
    gt_data = pd.read_csv(gt_file).values
    # Read the IMU CSV file; expected format: timestamp, wx, wy, wz, ax, ay, az
    imu_data = pd.read_csv(imu_file).values

    # Interpolate gyroscope data (columns 1:4) to match ground-truth timestamps
    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    # Interpolate accelerometer data (columns 4:7) to match ground-truth timestamps
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    
    # Extract position data from ground-truth (columns 1 to 3)
    pos_data = gt_data[:, 1:4]
    # Extract orientation data from ground-truth (columns 4 to 7; quaternion components)
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data

def load_oxiod_dataset(imu_file: str, gt_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the OXIOD dataset from CSV files and remove the first 1200 and last 300 samples.
    
    Args:
        imu_file: Path to the IMU CSV file.
        gt_file: Path to the ground-truth CSV file.
        
    Returns:
        Tuple containing:
         - gyro_data: Gyroscope data from the trimmed dataset.
         - acc_data: Accelerometer data from the trimmed dataset.
         - pos_data: Position data from ground-truth.
         - ori_data: Orientation data (quaternions) from ground-truth.
    """
    # Read the IMU and ground-truth CSV files
    imu_data = pd.read_csv(imu_file).values
    gt_data = pd.read_csv(gt_file).values

    # Remove the first 1200 and last 300 samples to trim the dataset
    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    # Extract gyroscope data from IMU (columns 4:7)
    gyro_data = imu_data[:, 4:7]
    # Extract accelerometer data from IMU (columns 10:13)
    acc_data = imu_data[:, 10:13]
    
    # Extract position data from ground-truth (columns 2:5)
    pos_data = gt_data[:, 2:5]
    # Orientation is stored as [w] then [x, y, z]; combine these columns accordingly:
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data

# ----------------------------
# Quaternion and Coordinate Utilities
# ----------------------------

def force_quaternion_uniqueness(q: quaternion.quaternion) -> quaternion.quaternion:
    """
    Ensure a unique quaternion representation by flipping its sign 
    if the first significant component is negative. This is important 
    because both q and -q represent the same rotation.
    
    Args:
        q: Input quaternion.
        
    Returns:
        Quaternion with a standardized, unique representation.
    """
    # Convert the quaternion to a float array representation
    q_data = quaternion.as_float_array(q)
    # Iterate through components to find the first one significantly non-zero
    for comp in q_data:
        if abs(comp) > 1e-5:  # Account for floating-point precision errors
            # If the first significant component is negative, flip the sign of the quaternion
            return -q if comp < 0 else q
    # If all components are near zero (edge case), return the quaternion as is
    return q

# ----------------------------
# Dataset Loader Functions
# ----------------------------

def load_dataset_6d_quat(gyro_data: np.ndarray, acc_data: np.ndarray, 
                           pos_data: np.ndarray, ori_data: np.ndarray, 
                           window_size: int = 200, stride: int = 10
                          ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load a 6D dataset with orientation represented as quaternions.
    
    This function creates sliding windows from the sensor data and computes:
      - Input windows for gyroscope and accelerometer readings.
      - Output windows representing changes in position (delta_p) and orientation (delta_q)
        between two offset points near the center of each window.
      - An initial pose (position and orientation) taken from the center of the first window.
    
    Returns:
        Tuple: ([x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q)
    """
    # Check if there is sufficient data to form a single window; if not, raise an error
    if gyro_data.shape[0] < window_size:
        raise ValueError("Not enough data to create even a single window.")

    # Determine the center index of the window (with a slight offset based on the stride)
    mid_idx = window_size // 2 - stride // 2
    # Set the initial position and orientation from the ground-truth data at the center of the first window
    init_p = pos_data[mid_idx, :]  # Initial position
    init_q = ori_data[mid_idx, :]  # Initial orientation (quaternion)

    # Initialize lists to store sliding window data for sensor inputs and target outputs
    x_gyro_windows, x_acc_windows = [], []
    y_delta_p_windows, y_delta_q_windows = [], []

    # Loop through the data using sliding window indices
    for idx in sliding_window_indices(gyro_data.shape[0], window_size, stride):
        # Extract a window for gyroscope data, skipping the first sample in the window
        x_gyro_windows.append(gyro_data[idx + 1: idx + 1 + window_size, :])
        # Extract a corresponding window for accelerometer data
        x_acc_windows.append(acc_data[idx + 1: idx + 1 + window_size, :])

        # Select two positions from near the center of the window for delta position computation
        p_a = pos_data[idx + window_size // 2 - stride // 2, :]  # First position
        p_b = pos_data[idx + window_size // 2 + stride // 2, :]  # Second position

        # Convert orientation data at the selected indices into quaternion objects
        q_a = quaternion.from_float_array(ori_data[idx + window_size // 2 - stride // 2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size // 2 + stride // 2, :])

        # Compute delta_p: transform the position difference into the local coordinate frame defined by q_a
        delta_p = (quaternion.as_rotation_matrix(q_a).T @ (p_b - p_a).T).T
        # Compute delta_q: the relative orientation change, ensuring a unique quaternion representation
        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        # Append the computed differences to the respective lists
        y_delta_p_windows.append(delta_p)
        y_delta_q_windows.append(quaternion.as_float_array(delta_q))

    # Stack the list of windows into arrays and return along with the initial pose
    return ([stack_windows(x_gyro_windows), stack_windows(x_acc_windows)],
            [stack_windows(y_delta_p_windows), stack_windows(y_delta_q_windows)],
            init_p,
            init_q)
