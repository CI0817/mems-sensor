This file contains the step to set up Raspberry Pi from scratch via CLI to run the necessary files to log data and run inference.

# 1. Setup custom udev rule for connection between Pi and motherboard
- Plug in Pi to motherboard
- Find its current tty: `dmesg | grep ttyACM`
- Use the following command: `udevadm info -a -n /dev/ttyACM0`
    - Replace ttyACM0 with the the last ttyACM* value from your previous step
- Note down the vendor and product ID, and serial number (if available)
- Create a new rule file in the `/etc/udev/rules.d/` directory: `sudo nano /etc/udev/rules.d/99-stm-vcp.rules`
- Paste the following line: `SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="stm_vcp"`
    - Replace the vendor and product ID with the actual IDs
    - The SYMLINK name can be anything, in this case it is "stm_vcp"
- Save the file and exit the editor
- Reload the `udev` rules: `sudo udevadm control --reload-rules`
- Replug the Pi and the motherboard
- Verify the new device name: `ls -l /dev/stm_vcp`

# 2. Create the appropriate directory structure
- Go to home directory: `cd ~`
- Make the rocket scripts directory: `mkdir rocket_scripts`
- Go into the new directory: `cd rocket_scripts`
- Create the sub-folders for the data to be logged:
```
mkdir all_imu_data
mkdir predicted_trajectory
```

# 3. Set up Python virtual environment
- Go into the desired directory vid `ls` and `cd`
- Create the virtual environment: `python3 -m venv .venv`
    - The name can be anything, in this case it's `.venv`
- Activate the virtual environment: `source .venv/bin/activate`
- Install the appropriate libraries (you need to have accesss to the internet first)
```
pip install pyserial
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
    - Recommend purging cache first: `pip cache purge`

# 4. SCP the trained model into Pi
- On your laptop, choose the directory of the file you want to transfer to the Pi: `cd /c/Users/chhay/rocket_scripts`
- Run SCP to transfer the trained model into the appropriate directory on the Pi: `scp train_model.pt chhay@pi402.local:~/rocket_scripts/`

# 5. Add all the required scripts
- `prepare-idle-state.sh`
    - Create the file on the Pi: `nano prepare-idle-state.sh`
    - Copy and paste the script in this directory into the CLI of the Pi
    - Save the file
    - Make the file executable: `chmod +x prepare-idle-state.sh`
- `listener.py`
    - Create the file on the Pi: `nano listener.py`
    - Copy and paste the script in this directory into the CLI of the Pi
    - Save the file
- `read_data.py`
    - Create the file on the Pi: `nano read_data.py`
    - Copy and paste the script in this directory into the CLI of the Pi
    - Save the file
- `run_inference.py`
    - Create the file on the Pi: `nano run_inference.py`
    - Copy and paste the script in this directory into the CLI of the Pi
    - Save the file
- `main.py`
    - Create the file on the Pi: `nano main.py`
    - Copy and paste the script in this directory into the CLI of the Pi
    - Save the file

# 6. Create the listener service file
- Create the service file: `sudo nano /etc/systemd/system/rocket-listener.service`
    - In this case, we choose the name rocket_listener.service
- Copy and paste the following content:
```
[Unit]
Description=Rocket Launch Listener Service
Wants=network.target
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/home/chhay/rocket_scripts
ExecStartPre=/home/chhay/rocket_scripts/prepare-idle-state.sh
ExecStart=/home/chhay/rocket_scripts/.venv/bin/python3 /home/chhay/rocket_scripts/listener.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

# 7. Enable and start the service file
- Reload `systemd` to make it aware of the new service file: `sudo systemctl daemon-reload`
- Enable the service to the start on every reboot: `sudo systemctl start rocket-listener.service`
- Check the status of the service: `sudo systemctl status rocket-listener.service`
- View live logs of the service: `sudo journalctl -u rocket-listener.service -f`
