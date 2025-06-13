Commands for communications between Pi and laptop

imu_app.service: service script to run inference and log data automatically upon boot
~/Desktop/code: directory containing all the scripts and venv to log data and run inference

### Enter SSH comm between pi and laptop
ssh chhay@pi401.local
password: 0209

ssh chhay@192.168.5.3
password: 0209
    
### Reboot the Pi (Pi)
sudo reboot

### Configure settings (Pi)
sudo raspi-config

### Check permission for file or folder (Pi)
ls -ld path/to/folder/or/file

### Enable/disable imu_app.service (Pi)
sudo systemctl enable imu_app.service
sudo systemctl disable imu_app.service

### Check imu_app.service status (Pi)
sudo systemctl status imu_app.service

### Stop the imu_app.service (Pi)
sudo systemctl stop imu_app.service

### Send script directory from laptop to Pi and vice versa (laptop)
scp -r C:\Users\chhay\code chhay@pi401.local:~/Desktop

scp -r chhay@pi401.local:~/Desktop/code C:\Users\chhay\code

#### Via ethernet
scp -r C:\Users\chhay\code chhay@192.168.5.3:~/Desktop

scp -r chhay@192.168.5.3:~/Desktop/code C:\Users\chhay\code

### Send data from Pi to laptop (laptop)
scp -r chhay@pi401.local:~/Desktop/code/all_imu_data C:\Users\chhay\code

scp -r chhay@pi401.local:~/Desktop/code/predicted_trajectory C:\Users\chhay\code

#### Via ethernet
scp -r chhay@192.168.5.3:~/Desktop/code/all_imu_data C:\Users\chhay\code

scp -r chhay@192.168.5.3:~/Desktop/code/predicted_trajectory C:\Users\chhay\code

### Delete all CSV files from the chosen folder (Pi)
rm all_imu_data/* predicted_trajectory/*

### Enable flight mode before launch (Pi)
sudo touch /boot/flight_mode && sudo reboot

### Disable flight mode after recovery (Pi)
sudo rm /boot/flight_mode && sudo reboot

### Regenerate the key after re-imaging the Pi
ssh-keygen -R pi401.local

### Enter NetworkManager (Pi)
sudo nmtui

### Find all network interfaces connected to the Pi (Pi)
nmcli device status

### Check the current power govenor on the Pi (Pi)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

### Monitor CPU usage (Pi)
htop

### Measure temperature, core voltage, clock speed (Pi)
vcgencmd measure_temp
vcgencmd measure_volts core
vcgencmd measure_clock arm