# read_data.py
import serial
import csv
import time
import threading
from queue import Queue, Full
import numpy as np
import os
import sys

# Constants (keep these)
MG_TO_G = 1 / 1000.0
MDPS_TO_DPS = 1 / 1000.0
G_TO_MS2 = 9.80665
DPS_TO_RADS = np.pi / 180.0

# --- Configuration ---
SERIAL_PORT = '/dev/stm_vcp' # IMPORTANT: Ensure this matches listener.py!
BAUD_RATE = 115200
CHOSEN_IMU_INDEX = 0 # Which IMU to use for inference (0-3)
NUM_IMUS = 4
EXPECTED_FIELDS = 1 + NUM_IMUS * 6
# cpu_boosted = False # Removed - listener handles CPU governor now
# --- End Configuration ---

# Function accepts stop_event and the queue object
def read_serial_data(port, baud, output_csv_filename, imu_index_for_inference, queue_obj, stop_event):
    """
    Reads serial data, logs all IMU data, and puts the chosen IMU's data
    onto the inference queue as it arrives. Assumes PCB controls data rate.
    Includes retry logic for opening the serial port.
    """
    ser = None # Initialize ser to None

    # --- Serial Connection Retry Logic ---
    max_retries = 10  # Try up to 10 times
    retry_delay_seconds = 3 # Wait 3 seconds between retries

    for attempt in range(max_retries):
        if stop_event.is_set(): # Check if stop requested during retries
            print("Read_data: Stop event detected during serial connection attempt.")
            sys.stdout.flush()
            return # Exit the function if stop requested

        try:
            print(f"Read_data: Attempting to open serial port {port} (Attempt {attempt + 1}/{max_retries})...")
            sys.stdout.flush()
            # Use a timeout for opening
            ser = serial.Serial(port, baud, timeout=1)
            print(f"Read_data: Serial port {port} opened successfully.")
            sys.stdout.flush()
            break # Exit loop if successful

        except serial.SerialException as e:
            print(f"Read_data: Attempt {attempt + 1} failed: {e}")
            sys.stdout.flush()
            if attempt < max_retries - 1:
                print(f"Read_data: Retrying in {retry_delay_seconds} seconds...")
                sys.stdout.flush()
                # Wait, but check stop_event frequently
                stop_event.wait(timeout=retry_delay_seconds)
            else:
                print("Read_data: Max retries reached. Could not open serial port.")
                sys.stdout.flush()
                stop_event.set() # Signal failure to other threads
                return # Exit function

        except Exception as e:
             print(f"Read_data: Unexpected error opening serial port: {e}")
             sys.stdout.flush()
             stop_event.set()
             return # Exit function

    # --- If Serial Port Opened Successfully ---
    if ser is None or not ser.is_open:
        print("Read_data: CRITICAL - Serial port not open after retry loop. Exiting thread.")
        sys.stdout.flush()
        stop_event.set()
        return # Exit if port couldn't be opened

    # --- Prepare CSV ---
    csv_header = ['timestamp_ms']
    for i in range(NUM_IMUS):
        csv_header.extend([
            f'imu{i}_acc_x_g', f'imu{i}_acc_y_g', f'imu{i}_acc_z_g',
            f'imu{i}_gyro_x_dps', f'imu{i}_gyro_y_dps', f'imu{i}_gyro_z_dps'
        ])

    # --- Main Reading Loop ---
    try:
        with open(output_csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)
            print(f"Read_data: Opened {output_csv_filename} for writing all IMU data.")
            sys.stdout.flush()

            # Start reading data from the serial port
            print("Read_data: Starting to read serial data...")
            while not stop_event.is_set():
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if not line: continue
                        parts = line.split(',')
                        if len(parts) != EXPECTED_FIELDS:
                            print(f"Read_data Warn: Malformed line. Expected {EXPECTED_FIELDS}, got {len(parts)}. Line: '{line}'")
                            sys.stdout.flush()
                            continue

                        timestamp_ms = int(parts[0])
                        csv_row = [timestamp_ms]
                        inference_data_for_queue = None # Data in m/s^2, rad/s for queue

                        for i in range(NUM_IMUS):
                            start_index = 1 + i * 6
                            try:
                                ax_mg = int(parts[start_index])
                                ay_mg = int(parts[start_index + 1])
                                az_mg = int(parts[start_index + 2])
                                gx_mdps = int(parts[start_index + 3])
                                gy_mdps = int(parts[start_index + 4])
                                gz_mdps = int(parts[start_index + 5])

                                # Convert for CSV logging (g and deg/s)
                                ax_g = ax_mg * MG_TO_G
                                ay_g = ay_mg * MG_TO_G
                                az_g = az_mg * MG_TO_G
                                gx_dps = gx_mdps * MDPS_TO_DPS
                                gy_dps = gy_mdps * MDPS_TO_DPS
                                gz_dps = gz_mdps * MDPS_TO_DPS
                                csv_row.extend([ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps])

                                # If this is the IMU for inference, prepare queue data
                                if i == imu_index_for_inference:
                                    # Convert units for the inference queue (m/s^2, rad/s)
                                    ax_ms2 = ax_g * G_TO_MS2
                                    ay_ms2 = ay_g * G_TO_MS2
                                    az_ms2 = az_g * G_TO_MS2
                                    gx_rads = gx_dps * DPS_TO_RADS
                                    gy_rads = gy_dps * DPS_TO_RADS
                                    gz_rads = gz_dps * DPS_TO_RADS
                                    inference_data_for_queue = [ax_ms2, ay_ms2, az_ms2, gx_rads, gy_rads, gz_rads]

                            except (ValueError, IndexError) as e:
                                print(f"Read_data Warn: Error parsing data for IMU {i} in line '{line}'. Error: {e}")
                                sys.stdout.flush()
                                csv_row.extend([0.0] * 6)
                                if i == imu_index_for_inference:
                                     inference_data_for_queue = None
                                break # Stop processing this line if parsing failed for one IMU

                        # --- Write to CSV ---
                        # Only write if parsing succeeded for all IMUs (or adjust logic if partial write ok)
                        if len(csv_row) == len(csv_header):
                             csv_writer.writerow(csv_row)

                        # --- Put Data on Queue ---
                        if inference_data_for_queue is not None:
                            try:
                                # Use the passed queue_obj
                                queue_obj.put((timestamp_ms, inference_data_for_queue), block=False)
                            except Full:
                                print("Read_data Warn: Inference queue is full. Dropping data.")
                                sys.stdout.flush()
                        elif i == imu_index_for_inference: # Only warn if the target IMU failed parsing
                             print(f"Read_data Warn: Skipping putting data to inference queue due to parsing error.")
                             sys.stdout.flush()

                    except UnicodeDecodeError:
                        print("Read_data Warn: Could not decode bytes to UTF-8. Skipping line.")
                        sys.stdout.flush()
                    except ValueError:
                        print(f"Read_data Warn: Could not parse numbers in line: '{line}'. Skipping.")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"Read_data: An unexpected error occurred in read loop: {e}")
                        sys.stdout.flush()
                        time.sleep(0.1) # Avoid busy-looping on unexpected errors
                else:
                    # If no data is waiting, check if the port is still physically connected
                    # This might help detect unplug events, although reliability varies
                    try:
                        if not ser.is_open: # Check if somehow closed
                             raise serial.SerialException("Serial port is not open.")
                        # Optional: A quick check like reading CTS might sometimes detect disconnects
                        # ser.getCTS()
                    except serial.SerialException as e:
                        print(f"Read_data: Serial port check failed: {e}. Assuming disconnect.")
                        sys.stdout.flush()
                        stop_event.set() # Signal other threads
                        break # Exit the while loop
                    except Exception: # Ignore other errors from checks
                        pass
                    time.sleep(0.001) # Wait briefly if no serial data

    except KeyboardInterrupt:
        print("Read_data: Keyboard interrupt detected.")
        sys.stdout.flush()
    except Exception as e:
         print(f"Read_data: An error occurred during serial processing or file writing: {e}")
         sys.stdout.flush()
    finally:
        # Ensure port is closed if it was successfully opened
        if ser and ser.is_open:
            ser.close()
            print("Read_data: Serial port closed.")
            sys.stdout.flush()
        # Ensure stop_event is set so other threads know this one finished/failed
        stop_event.set()
        print("Read_data: Read_serial_data thread finished.")
        sys.stdout.flush()

# Note: The __main__ block from the original read_data.py is removed
# as this script is now intended to be imported and run as a thread by main.py