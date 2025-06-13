import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_total_acceleration_subplots(csv_filepath, x_min_sec=None, x_max_sec=None):
    """
    Reads IMU data from a CSV file, calculates total acceleration for each
    sensor, and plots each on a separate subplot within a specified time range.

    Args:
        csv_filepath (str): The full path to the CSV file.
        x_min_sec (float, optional): The minimum time in seconds for the x-axis. Defaults to None.
        x_max_sec (float, optional): The maximum time in seconds for the x-axis. Defaults to None.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        # We'll use the first column 'timestamp_ms' as the index.
        imu_data = pd.read_csv(csv_filepath, index_col='timestamp_ms')
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # --- Time Calculation ---
    # Convert the timestamp from milliseconds to seconds and offset it
    # so the plot starts at 0 seconds.
    time_seconds = (imu_data.index - imu_data.index[0]) / 1000.0

    # --- Acceleration Calculation & Plotting Setup ---
    # We will create a 2x2 grid of subplots.
    # `figsize` is adjusted for a better layout of the 2x2 grid.
    # `sharex=True` and `sharey=True` make all subplots share the same x and y axes.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex=True, sharey=True)
    
    # Flatten the 2x2 array of axes for easy iteration
    axes = axes.flatten()

    plt.style.use('seaborn-v0_8-whitegrid') # Using a nice style for the plot

    # --- Loop, Calculate, and Plot ---
    # We'll now calculate the magnitude of the acceleration for each sensor
    # and plot it on its dedicated subplot.
    for i in range(4):
        accel_x = imu_data[f'imu{i}_acc_x_g']
        accel_y = imu_data[f'imu{i}_acc_y_g']
        accel_z = imu_data[f'imu{i}_acc_z_g']
        
        # Calculate the total acceleration (magnitude)
        total_accel = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Plotting on the respective subplot
        ax = axes[i]
        ax.plot(time_seconds, total_accel, label=f'IMU {i} Data')
        
        # --- Formatting each subplot ---
        ax.set_title(f'IMU {i} Total Acceleration', fontsize=14)
        ax.legend()
        ax.grid(True)

    # --- Global Plot Formatting ---
    # Add a single, overarching title for the entire figure
    fig.suptitle('Total Acceleration vs. Time for all IMU Sensors', fontsize=18, fontweight='bold')
    
    # Add shared labels for the x and y axes
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Total Acceleration (g)', ha='center', va='center', rotation='vertical', fontsize=12)

    # Set the x-axis (time) limits for all subplots.
    # Since sharex=True, we only need to set it on one of them.
    # `plt.xlim` gracefully handles None, so we don't need extra checks.
    plt.xlim(x_min_sec, x_max_sec)

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95]) # Adjust layout to make room for titles and labels
    plt.show()


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace this with the actual path to your CSV file.
    file_path = 'all_imu_data/all_imu_data_20250429_052908.csv'
    
    # Call the function to generate the plot.
    # You can now specify the time range in seconds.
    # For your example of 0 to 7500 ms, that is 0 to 7.5 seconds.
    # To view the whole timeline, you can just call: plot_total_acceleration_subplots(file_path)
    plot_total_acceleration_subplots(file_path, x_min_sec=0, x_max_sec=8.0)
