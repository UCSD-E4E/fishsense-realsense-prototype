#!/bin/bash

echo 388 > /sys/class/gpio/export
echo 298 > /sys/class/gpio/export
echo in > /sys/class/gpio/gpio388/direction
echo out > /sys/class/gpio/gpio298/direction
echo 1 > /sys/class/gpio/gpio298/active_low
/home/fishsensetx2/rs_save &>> /home/fishsensetx2/log.txt
