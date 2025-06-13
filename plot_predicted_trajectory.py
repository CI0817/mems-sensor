import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_dummy_data(filepath):
    """
    Generates a dummy CSV file with a plausible rocket trajectory.
    This version generates some negative position values to test the abs() correction.
    
    Args:
        filepath (str): The path where the dummy CSV will be saved.
    """
    print("Generating dummy data file...")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Parameters for a simple parabolic flight
    time_steps = 200
    total_time_s = 100
    g = -9.81  # gravity
    initial_velocity_z = 100
    
    # Timestamps in milliseconds
    timestamps = np.linspace(0, total_time_s * 1000, time_steps)
    
    # Position data - now includes negative values
    time_s = timestamps / 1000
    pos_x = 10 * time_s - 500  # Will be negative for the first half
    pos_y = 8 * time_s - 150   # Will be negative initially
    pos_z = initial_velocity_z * time_s + 0.5 * g * time_s**2
    
    # Orientation data (quaternions for a pitch-up and roll)
    pitch_angle = np.pi / 2 - (time_s/total_time_s * np.pi/2) # Pitch up
    roll_angle = np.pi / 4 * np.sin(time_s * 0.1) # Slow roll
    yaw_angle = np.pi/6 * (time_s/total_time_s) # Slow yaw
    
    # This is a simplified combination, for real scenarios use proper quaternion multiplication
    qw_p, qx_p, qy_p, qz_p = np.cos(pitch_angle/2), 0, np.sin(pitch_angle/2), 0
    qw_r, qx_r, qy_r, qz_r = np.cos(roll_angle/2), np.sin(roll_angle/2), 0, 0
    qw_y, qx_y, qy_y, qz_y = np.cos(yaw_angle/2), 0, 0, np.sin(yaw_angle/2)

    # Simple combination for dummy data (not a real multiplication)
    quat_w = qw_p * qw_r * qw_y
    quat_x = qx_r
    quat_y = qy_p
    quat_z = qz_y
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp_ms': timestamps.astype(np.int64),
        'pos_x': pos_x,
        'pos_y': pos_y,
        'pos_z': pos_z,
        'quat_w': quat_w,
        'quat_x': quat_x,
        'quat_y': quat_y,
        'quat_z': quat_z,
    })
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Dummy data saved to '{filepath}'")

def quaternion_to_euler(w, x, y, z):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw) in degrees.
    roll: rotation around the x-axis
    pitch: rotation around the y-axis
    yaw: rotation around the z-axis
    """
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0) # Clamp the value to avoid errors with arcsin
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Convert to degrees
    return np.degrees(roll_x), np.degrees(pitch_y), np.degrees(yaw_z)

def plot_rocket_trajectory(file_path, start_time_s=None, end_time_s=None):
    """
    Reads rocket trajectory data, clips it to a time range, and plots its 
    position and orientation with separate subplots for roll, pitch, and yaw.
    
    Args:
        file_path (str): The path to the CSV file.
        start_time_s (float, optional): The start time in seconds to clip the data. Defaults to None.
        end_time_s (float, optional): The end time in seconds to clip the data. Defaults to None.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the file exists or generate the dummy data.")
        return

    # --- Data Processing ---
    required_cols = ['timestamp_ms', 'pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: The CSV file is missing required columns. Expected: {required_cols}")
        return
    
    data['pos_x'] = data['pos_x'].abs()
    data['pos_y'] = data['pos_y'].abs()
    data['pos_z'] = data['pos_z'].abs()
    initial_timestamp = data['timestamp_ms'].iloc[0]
    data['time_s'] = (data['timestamp_ms'] - initial_timestamp) / 1000.0
    
    euler_angles = data.apply(
        lambda row: quaternion_to_euler(row['quat_w'], row['quat_x'], row['quat_y'], row['quat_z']),
        axis=1,
        result_type='expand'
    )
    euler_angles.columns = ['roll_deg', 'pitch_deg', 'yaw_deg']
    data = pd.concat([data, euler_angles], axis=1)

    # --- Time Clipping ---
    # Filter the data based on the provided time range before plotting
    if start_time_s is not None:
        data = data[data['time_s'] >= start_time_s]
    if end_time_s is not None:
        data = data[data['time_s'] <= end_time_s]

    if data.empty:
        print("Warning: No data available in the specified time range. Nothing to plot.")
        return

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Rocket Flight Analysis', fontsize=20, y=0.96)

    # Subplot 1: 3D Position Trajectory (top left)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(data['pos_x'], data['pos_y'], data['pos_z'], label='Trajectory', color='b', lw=2)
    ax1.scatter(data['pos_x'].iloc[0], data['pos_y'].iloc[0], data['pos_z'].iloc[0], color='g', s=100, label='Start', depthshade=False)
    ax1.scatter(data['pos_x'].iloc[-1], data['pos_y'].iloc[-1], data['pos_z'].iloc[-1], color='r', s=100, label='End', depthshade=False)
    ax1.set_title('3D Position (Absolute Values)', fontsize=14)
    ax1.set_xlabel('Position X (m)')
    ax1.set_ylabel('Position Y (m)')
    ax1.set_zlabel('Position Z (m) / Altitude')
    ax1.legend()
    ax1.grid(True)
    ax1.view_init(elev=20., azim=-135)

    # Subplot 2: Roll vs. Time (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(data['time_s'], data['roll_deg'], label='Roll', color='r')
    ax2.set_title('Roll vs. Time', fontsize=14)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.legend()
    ax2.grid(True)

    # Subplot 3: Pitch vs. Time (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(data['time_s'], data['pitch_deg'], label='Pitch', color='g')
    ax3.set_title('Pitch vs. Time', fontsize=14)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.legend()
    ax3.grid(True)

    # Subplot 4: Yaw vs. Time (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(data['time_s'], data['yaw_deg'], label='Yaw', color='b')
    ax4.set_title('Yaw vs. Time', fontsize=14)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Angle (degrees)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    file_path = 'predicted_trajectory/predicted_trajectory_20250429_052908.csv'
    
    # --- Set your desired time range for clipping here (in seconds) ---
    # Use 'None' to disable clipping on one end (e.g., start_time_s=None plots from the very beginning)
    # The dummy data runs for 100s, so a range like 10-80s is good for testing.
    # You can change this to 0 and 8000 for your actual data.
    start_time = 10
    end_time = 8000

    if not os.path.exists(file_path):
        print(f"Could not find '{file_path}'.")
        generate_dummy_data(file_path)
    
    # Run the plotting function with the specified time range
    plot_rocket_trajectory(file_path, start_time_s=None, end_time_s=None)
