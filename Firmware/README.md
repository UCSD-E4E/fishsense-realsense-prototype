# Installation Instructions
1. Obtain the following materiel
    1. FishSense Jetson flashing computer
    2. HDMI Monitor
    3. 12V Power Supply
    4. Keyboard
    5. Ethernet
    6. USB A to micro B cable
    7. Jetson TX2 with Orbitty Carrier Board
2. Connect the keyboard and monitor to the Jetson flashing computer
3. Connect the Orbitty Carrier Board to the Jetson flashing computer via the USB A to micro B cable
4. Connect the Orbitty Carrier Board to the 12V power supply
5. Ensure that both switches on S1 are in the OFF position (away from the 12V connector)
6. Power on the 12V power supply
7. Boot the Jetson into the recovery mode by pressing and holding RECOVERY, RESET, and POWER, then releasing POWER, RESET, and RECOVERY in that exact order.
8. On the Jetson flashing computer, open a terminal and execute the following commands:
    1. `cd ~/nvidia/nvidia_sdk/JetPack_4.5.1_Linux_JETSON_TX2/Linux_for_Tegra/`
    2. `sudo ./flash.sh ./cti/tx2/orbitty mmcblk0p1`
9. Wait for the above commands to complete.
10. Plug the monitor, keyboard, and ethernet into the Orbitty Carrier Board.
11. Press the RESET button on the Orbitty Carrier Board
12. Follow the onscreen instructions to configure the Jetson.  Use the following settings:
    1. Hostname: `e4e-fishsense-XXX`, where XXX is the serial number of the system
    2. Username: `e4e`
    3. Password: `stingray`
    4. No wifi
13. Once the Jetson has booted to the desktop, open a terminal and execute the following commands:
    1. `sudo apt-get update`
    2. `sudo apt-get install -y -f`
    3. `sudo apt-get upgrade`
    4. `sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE`
    5. `sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u`
    6. `sudo apt-get install -y tmux git vim htop glances build-essential librealsense2-utils librealsense2-dev librealsense2-dbg`
    7. `git clone https://github.com/UCSD-E4E/FishSense`
    8. `cd FishSense/Firmware`
    9. `make`
    10. `sudo make install`
14. Ensure that both switches on S1 are in the ON position (closer to the 12V connector)
