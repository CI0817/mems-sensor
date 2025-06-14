import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory_and_orientation(csv_filepath):
    """
    Reads trajectory data from a CSV, plots the 3D position with a time-based
    color gradient, and plots the orientation (as Euler angles) in separate subplots.

    Args:
        csv_filepath (str): The full path to the CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame.
        # We assume the header row is correct as specified.
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        print("Please ensure the file path is correct and the script is run from the correct directory.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # --- Time Calculation ---
    # Convert the timestamp from milliseconds to seconds and offset it
    # so the plot starts at 0 seconds. This makes the time axis more readable.
    time_seconds = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000.0

    # --- Figure 1: 3D Position Plot ---
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Create the 3D scatter plot with a color map.
    # The 'c' argument maps each point's value in 'time_seconds' to a color.
    # 'cmap' specifies which color map to use; 'viridis' is a good-looking, perceptually uniform choice.
    sc = ax1.scatter(df['pos_x'], df['pos_y'], df['pos_z'], c=time_seconds, cmap='viridis', marker='o', s=5)
    
    # Add a color bar to serve as a legend for the time-to-color mapping.
    cbar = plt.colorbar(sc, shrink=0.5, aspect=10)
    cbar.set_label('Time (seconds)', fontsize=12)
    
    # Formatting the 3D plot
    ax1.set_title('3D Predicted Trajectory', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Position X (m)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Position Y (m)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Position Z (m)', fontsize=12, labelpad=10)
    ax1.grid(True)
    
    # --- Figure 2: Orientation (Euler Angles) Plot ---
    
    # --- Quaternion to Euler Conversion ---
    # This is the standard set of equations to convert a quaternion to Euler angles.
    w, x, y, z = df['quat_w'], df['quat_x'], df['quat_y'], df['quat_z']

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_rad = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    # Ensure the value is within the valid domain for arcsin [-1, 1] to avoid NaN errors
    # due to floating point inaccuracies. This is a robust programming practice.
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_rad = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_rad = np.arctan2(t3, t4)
    
    # Convert radians to degrees for easier interpretation on the plots.
    roll_deg = np.rad2deg(roll_rad)
    pitch_deg = np.rad2deg(pitch_rad)
    yaw_deg = np.rad2deg(yaw_rad)

    # --- Plotting Euler Angles ---
    # Create a new figure with 3 subplots stacked vertically, sharing the same x-axis.
    fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 9), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Roll
    axes[0].plot(time_seconds, roll_deg, label='Roll', color='crimson')
    axes[0].set_title('Roll', fontsize=14)
    axes[0].set_ylabel('Degrees')
    axes[0].grid(True)

    # Plot Pitch
    axes[1].plot(time_seconds, pitch_deg, label='Pitch', color='mediumseagreen')
    axes[1].set_title('Pitch', fontsize=14)
    axes[1].set_ylabel('Degrees')
    axes[1].grid(True)

    # Plot Yaw
    axes[2].plot(time_seconds, yaw_deg, label='Yaw', color='royalblue')
    axes[2].set_title('Yaw', fontsize=14)
    axes[2].set_ylabel('Degrees')
    axes[2].grid(True)
    
    # --- Global Formatting for Figure 2 ---
    fig2.suptitle('Orientation (Euler Angles) vs. Time', fontsize=18, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == '__main__':
    file_path = 'data\irec_launch_120625\predicted_trajectory\predicted_trajectory_20250429_052908.csv'
    
    # Call the function to generate the plots.
    plot_trajectory_and_orientation(file_path)
    
    # This single call will display all figures that have been created.
    plt.show()
