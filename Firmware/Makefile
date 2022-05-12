rs_save: rs_save.cpp
	g++ rs_save.cpp -o rs_save -lrealsense2
.phony: install

install: rs_save
	-systemctl stop savescript
	-systemctl disable savescript
	mkdir -p /usr/local/share/FishSense
	mkdir -p /mnt/data
	if [ `cat /etc/fstab | grep /mnt/data | wc -l` = 0 ] ; then\
		echo "/dev/sda1 /mnt/data exfat defaults 0 0" >> /etc/fstab;\
	fi
	cp rs_save /usr/bin/rs_save
	cp run_firmware.sh /usr/bin/run_fs_firmware.sh
	chmod +x /usr/bin/run_fs_firmware.sh
	cp savescript.service /lib/systemd/system/savescript.service
	cp savescript.service /etc/systemd/system/savescript.service
	chmod 644 /etc/systemd/system/savescript.service
	systemctl enable savescript
	systemctl start savescript	
