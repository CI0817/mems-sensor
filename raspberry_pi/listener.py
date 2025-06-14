import serial
import subprocess
import time
import sys
import os # Needed for os.execv

# --- !!! CONFIGURATION - EDIT THESE !!! ---
SERIAL_PORT = '/dev/stm_vcp'
BAUD_RATE = 115200
MAIN_APP_PATH = '/home/chhay/rocket_scripts/main.py'

# --- Serial Connection Retry Logic ---
max_retries = 10  # Try up to 10 times
retry_delay_seconds = 3 # Wait 3 seconds between retries
ser = None

print("Rocket Listener: Starting up...")
sys.stdout.flush()

for attempt in range(max_retries):
    try:
        print(f"Rocket Listener: Attempting to open serial port {SERIAL_PORT} (Attempt {attempt + 1}/{max_retries})...")
        sys.stdout.flush()
        # Try opening the port (use a timeout for opening itself)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Rocket Listener: Serial port {SERIAL_PORT} opened successfully.")
        sys.stdout.flush()
        break # Exit loop if connection is successful

    except serial.SerialException as e:
        print(f"Rocket Listener: Attempt {attempt + 1} failed: {e}")
        sys.stdout.flush()
        if attempt < max_retries - 1:
            print(f"Rocket Listener: Retrying in {retry_delay_seconds} seconds...")
            sys.stdout.flush()
            time.sleep(retry_delay_seconds) # Simple sleep, no stop_event here
        else:
            print("Rocket Listener: Max retries reached. Could not open serial port.")
            sys.stdout.flush()
            # Exit the script if we can't open the port - systemd will handle restart
            sys.exit(1)
    except Exception as e:
        # Catch any other unexpected error during opening
        print(f"Rocket Listener: Unexpected error opening serial port: {e}")
        sys.stdout.flush()
        sys.exit(1)


# --- If Serial Port Opened Successfully ---
if ser and ser.is_open:
    print(f"Rocket Listener: Waiting for data on {SERIAL_PORT}...")
    sys.stdout.flush()
    try:
        # Wait indefinitely for the first byte (using the opened 'ser' object)
        # Set a long timeout for the read itself, or None to block forever
        ser.timeout = None
        data = ser.read(1)
        print(f"Rocket Listener: Launch trigger detected! Received: {data}")
        sys.stdout.flush()

        # --- LAUNCH ACTIONS ---
        print("Rocket Listener: Setting CPU governor to performance...")
        sys.stdout.flush()
        # Ensure path matches 'which cpufreq-set' output (/usr/bin/cpufreq-set)
        subprocess.run(['sudo', '/usr/bin/cpufreq-set', '-g', 'performance'], check=True)

        print(f"Rocket Listener: Executing main application: {MAIN_APP_PATH}")
        sys.stdout.flush()

        # Close the serial port *before* execv replaces the process
        # (Good practice, although execv would close file descriptors anyway)
        ser.close()
        print("Rocket Listener: Serial port closed.")
        sys.stdout.flush()

        # Replace the listener process with the main application process
        # Use the venv python interpreter
        os.execv('/home/chhay/rocket_scripts/.venv/bin/python3', ['/home/chhay/rocket_scripts/.venv/bin/python3', MAIN_APP_PATH])
        # Code below this line will not execute if execv is successful

    except serial.SerialException as e:
        print(f"Rocket Listener: Serial Error during read - {e}", file=sys.stderr)
        sys.stderr.flush()
        if ser and ser.is_open: ser.close()
        sys.exit(1) # Exit if serial fails during read
    except subprocess.CalledProcessError as e:
        print(f"Rocket Listener: Subprocess Error (likely cpufreq-set) - {e}", file=sys.stderr)
        sys.stderr.flush()
        if ser and ser.is_open: ser.close()
        sys.exit(1) # Exit if command fails
    except Exception as e:
        print(f"Rocket Listener: Unexpected Error after opening port - {e}", file=sys.stderr)
        sys.stderr.flush()
        if ser and ser.is_open: ser.close()
        sys.exit(1) # Exit on other errors
    finally:
         # Ensure port is closed if an error happened before execv but after open
        if ser and ser.is_open:
             ser.close()
             print("Rocket Listener: Serial port closed in finally block.")
             sys.stdout.flush()
else:
     # This case should ideally be handled by sys.exit(1) in the loop, but as a failsafe:
     print("Rocket Listener: Script exiting because serial port was not opened.")
     sys.stdout.flush()
     sys.exit(1)

# This line should now be unreachable if execv succeeds
print("Rocket Listener: Listener script finished. (This shouldn't print if execv worked)")
sys.stdout.flush()