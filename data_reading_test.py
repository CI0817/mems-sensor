import serial
import time
import sys

# --- Configuration ---
SERIAL_PORT = '/dev/ttySTM32'  # <<< CHANGE THIS if your STM32 shows up on a different port
BAUD_RATE = 115200
READ_TIMEOUT = 1.0
RECONNECT_DELAY = 5.0
NUM_IMUS = 4                # <<< Number of IMUs
VALUES_PER_IMU = 6          # Accel(X,Y,Z) + Gyro(X,Y,Z)
EXPECTED_VALUES = 1 + (NUM_IMUS * VALUES_PER_IMU) # 1 Timestamp + 24 sensor values = 25

# --- Helper Function for Connection (Same as before) ---
def connect_serial(port, baud, timeout):
    """Attempts to connect to the serial port. Returns serial object or None."""
    try:
        print(f"Attempting to connect to {port}...")
        # Ensure correct permissions: Add user to 'dialout' group
        # sudo usermod -a -G dialout $USER
        # You might need to log out and back in after adding the user to the group.
        ser_conn = serial.Serial(port, baud, timeout=timeout)
        print(f"Connected successfully to {port}.")
        ser_conn.flushInput()  # Clear any old data received but not read
        return ser_conn
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e}")
        print("Check connection, port name, and permissions ('dialout' group).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        return None

# --- Main Program ---
ser = None
try:
    # Initial connection loop (Same as before)
    while ser is None:
        ser = connect_serial(SERIAL_PORT, BAUD_RATE, READ_TIMEOUT)
        if ser is None:
            print(f"Retrying connection in {RECONNECT_DELAY:.1f} seconds...")
            time.sleep(RECONNECT_DELAY)

    print("\n--- Starting Data Acquisition ---")
    # Main data reading loop
    while True:
        try:
            if not ser.is_open:
                 raise serial.SerialException("Serial port is not open.")

            if ser.in_waiting > 0:
                line_bytes = ser.readline()
                if not line_bytes: continue

                try:
                    line_str = line_bytes.decode('utf-8').strip()
                except UnicodeDecodeError:
                    print(f"Warning: Could not decode bytes: {line_bytes!r}")
                    continue
                if not line_str: continue

                # --- Parse the CSV data ---
                try:
                    values_str_list = line_str.split(',')
                    if len(values_str_list) == EXPECTED_VALUES:
                        # Convert all values from string to integer
                        values_int = [int(v) for v in values_str_list] # Raises ValueError if conversion fails

                        # --- Extract timestamp and ALL sensor data ---
                        timestamp_ms = values_int[0] # First value is timestamp
                        sensor_data_flat = values_int[1:] # Get all sensor values (list of 24 ints)

                        # --- Reshape and Convert Data ---
                        all_accel_g = []
                        all_gyro_dps = []
                        for i in range(NUM_IMUS):
                            # Calculate start and end indices for this IMU within sensor_data_flat
                            start_index = i * VALUES_PER_IMU
                            accel_start = start_index
                            gyro_start = start_index + 3

                            # Extract raw mg/mdps values for this IMU
                            accel_mg = sensor_data_flat[accel_start : accel_start + 3]
                            gyro_mdps = sensor_data_flat[gyro_start : gyro_start + 3]

                            # Convert to standard units (G's and dps) and store
                            all_accel_g.append([val / 1000.0 for val in accel_mg])
                            all_gyro_dps.append([val / 1000.0 for val in gyro_mdps])

                        # --- Print formatted output for ALL IMUs ---
                        print(f"Timestamp: {timestamp_ms:<10d} ms") # Print timestamp once
                        for i in range(NUM_IMUS):
                            accel_g = all_accel_g[i]
                            gyro_dps = all_gyro_dps[i]
                            print(f"  IMU[{i}] | Accel [G]: "
                                  f"X={accel_g[0]:+7.3f}, Y={accel_g[1]:+7.3f}, Z={accel_g[2]:+7.3f} | "
                                  f"Gyro [dps]: "
                                  f"X={gyro_dps[0]:+8.3f}, Y={gyro_dps[1]:+8.3f}, Z={gyro_dps[2]:+8.3f}")
                        # Optional: Add a separator line for better readability
                        print("-" * 80)


                        # *** YOUR CUSTOM DATA PROCESSING/LOGGING GOES HERE ***
                        # You now have:
                        # - timestamp_ms (int)
                        # - all_accel_g (list of 4 lists, each with 3 floats [G])
                        # - all_gyro_dps (list of 4 lists, each with 3 floats [dps])
                        # Example: log_data_to_file(timestamp_ms, all_accel_g, all_gyro_dps)

                    else:
                        print(f"Error: Expected {EXPECTED_VALUES} values, got {len(values_str_list)}. Line: '{line_str}'")

                except ValueError as ve:
                    print(f"Error: Cannot parse integer values in line: '{line_str}'. Details: {ve}")
                except Exception as parse_e:
                    print(f"Error parsing or processing data: {parse_e}")
            else:
                time.sleep(0.001) # Prevent busy-waiting if no data

        # ... (Exception handling: SerialException, KeyboardInterrupt, general Exception - Same as before) ...
        except serial.SerialException as e:
            print(f"\nSerial communication error: {e}")
            if ser and ser.is_open: ser.close()
            ser = None
            print("Attempting to reconnect...")
            while ser is None:
                 print(f"Retrying connection in {RECONNECT_DELAY:.1f} seconds...")
                 time.sleep(RECONNECT_DELAY)
                 ser = connect_serial(SERIAL_PORT, BAUD_RATE, READ_TIMEOUT)
            if ser: print("\n--- Connection re-established, resuming data acquisition ---")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting...")
            break
        except Exception as loop_e:
            print(f"\nAn unexpected error occurred: {loop_e}")
            time.sleep(1)

except Exception as critical_e:
     print(f"\nA critical error occurred outside the main loop: {critical_e}")

finally:
    # Ensure the serial port is closed upon exiting (Same as before)
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    print("Script terminated.")
    sys.exit()