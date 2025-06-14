# main.py
import threading
import time
import signal
import sys
from datetime import datetime
from queue import Queue
import os

# Import functions/variables from the other scripts
# Ensure these imports work correctly based on your file structure
from read_data import read_serial_data, SERIAL_PORT, BAUD_RATE, CHOSEN_IMU_INDEX
from run_inference import run_inference_loop, load_model_from_checkpoint, MODEL_CHECKPOINT, DEVICE

# Use a single stop event for both threads
stop_event = threading.Event()

# all imu data directory and predicted trajectory directory
ALL_IMU_DATA_DIR = "all_imu_data"
PREDICTED_TRAJECTORY_DIR = "predicted_trajectory"

def signal_handler(sig, frame):
    """Handles Ctrl+C interrupts gracefully."""
    print("\nCtrl+C detected! Signaling threads to stop...")
    sys.stdout.flush()
    stop_event.set()

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # --- Generate Timestamp for Filenames ---
    # Format: YYYYMMDD_HHMMSS (e.g., 20250414_134804)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure output directories exist (create if they don't) 
    if not os.path.exists(ALL_IMU_DATA_DIR):
        os.makedirs(ALL_IMU_DATA_DIR)
    if not os.path.exists(PREDICTED_TRAJECTORY_DIR):
        os.makedirs(PREDICTED_TRAJECTORY_DIR)

    output_csv_all_filename = f"{ALL_IMU_DATA_DIR}/all_imu_data_{timestamp_str}.csv"
    output_csv_predicted_filename = f"{PREDICTED_TRAJECTORY_DIR}/predicted_trajectory_{timestamp_str}.csv"

    print("--- Starting IMU Data Processing and Inference ---")
    print(f"Serial Port: {SERIAL_PORT}, Baud Rate: {BAUD_RATE}")
    print(f"Using IMU Index: {CHOSEN_IMU_INDEX} for inference")
    print(f"Model Checkpoint: {MODEL_CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print(f"Output CSV (All IMUs): {output_csv_all_filename}")
    print(f"Output CSV (Predicted Trajectory): {output_csv_predicted_filename}")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush() # Ensure initial messages appear

    # --- Load Model ---
    model = load_model_from_checkpoint(MODEL_CHECKPOINT, DEVICE)
    if model is None:
        print("Failed to load the model. Exiting.")
        sys.stdout.flush()
        sys.exit(1) # Exit if model loading fails

    # --- Create Shared Queue ---
    inference_queue_instance = Queue(maxsize=500)

    # --- Create Threads ---
    reader_thread = threading.Thread(
        target=read_serial_data,
        # Ensure args match def read_serial_data(port, baud, output_csv_filename, imu_index_for_inference, queue_obj, stop_event):
        args=(SERIAL_PORT, BAUD_RATE, output_csv_all_filename, CHOSEN_IMU_INDEX, inference_queue_instance, stop_event), # Passed queue and stop_event
        daemon=True # Allows main thread to exit even if this thread is running
    )

    inference_thread = threading.Thread(
        target=run_inference_loop,
        # Ensure args match def run_inference_loop(model, inference_queue, output_csv_filename, stop_event):
        args=(model, inference_queue_instance, output_csv_predicted_filename, stop_event), # Passed queue and stop_event
        daemon=True
    )

    # --- Start Threads ---
    print("Starting reader thread...")
    sys.stdout.flush()
    reader_thread.start()

    # Small delay to allow the reader to potentially connect before inference starts waiting
    time.sleep(1)

    print("Starting inference thread...")
    sys.stdout.flush()
    inference_thread.start()

    # --- Keep Main Thread Alive ---
    try:
        # Keep the main thread running while the worker threads are active
        while reader_thread.is_alive() or inference_thread.is_alive():
             # Check if stop was triggered externally (e.g., by a thread failing)
            if stop_event.is_set():
                 if reader_thread.is_alive() or inference_thread.is_alive():
                     print("Stop event detected in main loop. Waiting for threads...")
                     sys.stdout.flush()
                 break # Exit the monitoring loop if stop is signaled

            # Check if only one thread is left (might indicate the other crashed)
            # Optional: add more sophisticated monitoring here if needed

            time.sleep(0.5) # Check status periodically

    except KeyboardInterrupt:
         # This should be caught by the signal handler, but as a fallback:
         print("Keyboard interrupt caught in main loop.")
         sys.stdout.flush()
         if not stop_event.is_set():
              stop_event.set()

    # --- Wait for Threads to Finish (if not already stopped) ---
    print("Waiting for threads to finish...")
    sys.stdout.flush()
    # Join daemon threads with a timeout to prevent hanging
    reader_thread.join(timeout=2.0)
    inference_thread.join(timeout=2.0)

    if reader_thread.is_alive():
        print("Reader thread did not stop cleanly.")
        sys.stdout.flush()
    if inference_thread.is_alive():
        print("Inference thread did not stop cleanly.")
        sys.stdout.flush()

    print("--- Main script finished ---")
    sys.stdout.flush()