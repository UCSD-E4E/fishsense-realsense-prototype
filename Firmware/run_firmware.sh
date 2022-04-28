#!/bin/bash

echo 388 > /sys/class/gpio/export
echo 298 > /sys/class/gpio/export
echo in > /sys/class/gpio/gpio388/direction
echo out > /sys/class/gpio/gpio298/direction
echo 1 > /sys/class/gpio/gpio298/active_low
/usr/bin/rs_save &>> /var/log/rs_log.txt
