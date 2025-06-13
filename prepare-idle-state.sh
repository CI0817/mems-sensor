#!/bin/bash

# --- Flag File Path ---
DEV_MODE_FLAG="/home/chhay/ENABLE_DEV_MODE" # Using your home directory

echo "Preparing system for low-power idle state..."

if [ ! -f "$DEV_MODE_FLAG" ]; then
    # --- FLIGHT MODE ---
    echo "FLIGHT MODE: Entering maximum power saving mode."
    /usr/bin/cpufreq-set -g powersave
    /usr/sbin/rfkill block bluetooth
    /usr/sbin/rfkill block wifi

    echo "Forcing CPU to minimum frequency..."
    MIN_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        echo $MIN_FREQ > "$policy/scaling_max_freq"
    done

    echo "Disabling HDMI port..."
    /usr/bin/tvservice -o >/dev/null 2>&1

    echo "Disabling onboard LEDs..."
    echo 0 > /sys/class/leds/led0/brightness
    echo 0 > /sys/class/leds/led1/brightness

else
    # --- DEV MODE ---
    echo "DEV MODE: Entering development mode."
    /usr/bin/cpufreq-set -g ondemand
    /usr/sbin/rfkill unblock bluetooth
    /usr/sbin/rfkill unblock wifi
fi

echo "Attempting to enable USB autosuspend for STM32 device..."
for dev in /sys/bus/usb/devices/*; do
  if [ -f "$dev/idVendor" ] && [[ "$(cat $dev/idVendor)" == "0483" ]]; then
    echo "Found STM32 device at $dev. Enabling autosuspend."
    echo 'auto' > "$dev/power/control"
    echo 2000 > "$dev/power/autosuspend_delay_ms"
  fi
done

# Set ownership of the vcp port to the 'chhay' user if it exists
if [ -e /dev/stm_vcp ]; then
    sudo chown chhay:chhay /dev/stm_vcp
fi

echo "System state preparation complete."
exit 0