rs_save:
	g++ rs_save.cpp -o rs_save -lrealsense2
.phony: install

install: rs_save
	cp rs_save /usr/bin/rs_save
	cp run_firmware.sh /usr/bin/run_fs_firmware.sh
	chmod +x /usr/bin/run_fs_firmware.sh
	cp savescript.service /lib/systemd/system/savescript.service
	cp savescript.service /etc/systemd/system/savescript.service
	chmod 644 /etc/systemd/system/savescript.service
	systemctl start savescript	
