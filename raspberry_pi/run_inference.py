# run_inference.py
import time
import threading
import csv
import numpy as np
import pandas as pd
import torch
import quaternion
from queue import Queue, Empty

# Assuming model.py is in the same directory or accessible in PYTHONPATH
try:
    from training.initial_training.model import create_model
except ImportError:
    print("Error: Could not import 'create_model' from model.py.")
    print("Ensure model.py is in the correct path.")
    exit() # Or handle appropriately

# --- Configuration ---
MODEL_CHECKPOINT = 'trained_model.pt'
WINDOW_SIZE = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- End Configuration ---

# Buffer to hold incoming data for windowing
data_buffer = []

def load_model_from_checkpoint(checkpoint_path, device):
    """Loads the PyTorch model from a checkpoint file."""
    print(f"Loading model from: {checkpoint_path}")
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Determine model type from checkpoint if saved, else default (e.g., 'lstm')
        model_type = checkpoint.get('model_type', 'lstm') # Adjust default if needed
        print(f"Detected/assuming model type: {model_type}")
        model = create_model(model_type=model_type)
        # Load the model's learned parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # Move model to the specified device
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None

def run_inference_loop(model, inference_queue, output_csv_filename, stop_event):
    """
    Processes data from the queue, runs inference, and saves predictions.
    """
    global data_buffer
    poses = [] # List to store predicted poses: (position_np_array, quaternion_np_array)
    timestamps_output = [] # List to store timestamps corresponding to poses

    # Initial pose (assuming starting at origin with identity orientation)
    current_p = np.zeros(3)
    current_q_np = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z (identity quaternion)
    poses.append((current_p, current_q_np)) # Add initial pose
    initial_timestamp_ms = None

    print(f"Inference using device: {DEVICE}")
    print(f"Waiting for data... Inference will begin once {WINDOW_SIZE} samples are buffered.")

    # Prepare CSV file for predicted trajectory
    csv_header = [
        'timestamp_ms', 'pos_x', 'pos_y', 'pos_z',
        'quat_w', 'quat_x', 'quat_y', 'quat_z'
    ]
    try:
         with open(output_csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)
            # Write initial pose? Maybe wait until first prediction.
            print(f"Opened {output_csv_filename} for writing predicted trajectory.")

            while not stop_event.is_set():
                try:
                    # Get data from the queue (non-blocking)
                    timestamp_ms, imu_data_list = inference_queue.get(block=True, timeout=0.1) # Wait briefly if empty

                    # Store initial timestamp if not already set
                    if initial_timestamp_ms is None:
                        initial_timestamp_ms = timestamp_ms
                        timestamps_output.append(initial_timestamp_ms) # Timestamp for the initial pose
                        # Write initial pose to CSV
                        csv_writer.writerow([initial_timestamp_ms] + current_p.tolist() + current_q_np.tolist())


                    # Append new data sample (timestamp, [ax, ay, az, gx, gy, gz])
                    data_buffer.append((timestamp_ms, imu_data_list))

                    # Check if we have enough data for a window
                    if len(data_buffer) >= WINDOW_SIZE:
                        # Prepare the window
                        # Get the last WINDOW_SIZE samples
                        window_data = [item[1] for item in data_buffer[-WINDOW_SIZE:]]
                        # Get the timestamp corresponding to the *end* of this window
                        window_end_timestamp_ms = data_buffer[-1][0]

                        # Convert list of lists to numpy array
                        window_np = np.array(window_data, dtype=np.float32)

                        # Convert numpy array to torch tensor and move to device
                        # Input shape needs to be (batch_size, sequence_length, features)
                        # Here, batch_size=1, sequence_length=WINDOW_SIZE, features=6
                        gyro_tensor = torch.FloatTensor(window_np[:, 3:]).unsqueeze(0).to(DEVICE) # Gyro is last 3 columns
                        acc_tensor = torch.FloatTensor(window_np[:, :3]).unsqueeze(0).to(DEVICE)  # Accel is first 3 columns

                        # --- Run Model Inference ---
                        with torch.no_grad(): # Disable gradient calculations
                            delta_p_pred, delta_q_pred = model(gyro_tensor, acc_tensor)

                        # Convert predictions back to numpy arrays
                        delta_p_np = delta_p_pred.cpu().numpy()[0] # Remove batch dim
                        delta_q_np = delta_q_pred.cpu().numpy()[0] # Remove batch dim

                        # --- Integrate Pose ---
                        # Get the last calculated pose
                        last_p, last_q_np = poses[-1]

                        # Use numpy-quaternion for calculations
                        q_last = quaternion.from_float_array(last_q_np)
                        # Ensure predicted delta_q is a valid unit quaternion representation
                        q_delta = quaternion.from_float_array(delta_q_np)
                        # It's crucial that the model outputs normalized quaternions, or normalize here:
                        # q_delta = q_delta.normalized() # Uncomment if model doesn't guarantee normalization

                        # Update orientation: q_new = q_last * q_delta
                        current_q = q_last * q_delta
                        current_q_np = quaternion.as_float_array(current_q)

                        # Update position: p_new = p_last + R(q_last) * delta_p
                        # R(q_last) is the rotation matrix from the body frame to the world frame
                        rotation_matrix = quaternion.as_rotation_matrix(q_last)
                        # Transform delta_p (assumed to be in the body frame of q_last) to world frame and add
                        current_p = last_p + rotation_matrix @ delta_p_np

                        # Store the new pose and corresponding timestamp
                        poses.append((current_p, current_q_np))
                        timestamps_output.append(window_end_timestamp_ms)

                        # Write the new pose to CSV
                        csv_writer.writerow([window_end_timestamp_ms] + current_p.tolist() + current_q_np.tolist())

                        # --- Buffer Management ---
                        # Remove the oldest sample to maintain buffer size for sliding window
                        data_buffer.pop(0)

                except Empty:
                    # Queue was empty, just loop again
                    if stop_event.is_set(): # Check stop event again in case it was set while waiting
                         break
                    continue # Continue waiting for data
                except Exception as e:
                     print(f"An error occurred during inference or pose update: {e}")
                     time.sleep(0.1) # Avoid busy-looping

    except KeyboardInterrupt:
        print("Keyboard interrupt detected during inference.")
    except Exception as e:
        print(f"An error occurred in the inference loop: {e}")
    finally:
        stop_event.set() # Ensure other threads know to stop
        print("Run_inference_loop thread finished.")


if __name__ == '__main__':
    # This part is for testing the inference script independently
    # Requires a dummy queue and potentially a dummy reader thread or pre-filled queue

    print("--- Inference Standalone Test ---")
    # Load the model first
    model = load_model_from_checkpoint(MODEL_CHECKPOINT, DEVICE)

    if model:
        # Create a dummy queue for testing
        test_queue = Queue(maxsize=500)
        test_stop_event = threading.Event()

        # Example: Fill queue with some dummy data
        print("Populating test queue with dummy data...")
        start_time = time.time() * 1000
        for i in range(WINDOW_SIZE + 50): # Fill enough for a few windows
            dummy_ts = int(start_time + i * 10) # Simulate 100Hz
            # Dummy data (replace with more realistic values if needed)
            # [ax_ms2, ay_ms2, az_ms2, gx_rads, gy_rads, gz_rads]
            dummy_data = [0.1*i, 0.2, 9.8, 0.01*i, 0.02, 0.03]
            test_queue.put((dummy_ts, dummy_data))
            time.sleep(0.001) # Brief pause
        print(f"Test queue populated with {test_queue.qsize()} items.")


        # Start the inference loop in a thread
        print("Starting inference thread...")
        inference_thread = threading.Thread(target=run_inference_loop, args=(model, test_queue, OUTPUT_CSV_PREDICTED, test_stop_event))
        inference_thread.start()

        try:
            # Let it run for a bit
            time.sleep(10) # Run for 10 seconds
        except KeyboardInterrupt:
            print("Main thread interrupted.")
        finally:
            print("Stopping inference thread...")
            test_stop_event.set() # Signal the thread to stop

        inference_thread.join() # Wait for the thread to finish
        print("Inference standalone test finished.")
    else:
        print("Could not load model. Aborting test.")