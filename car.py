#!/usr/bin/env python3

import termios
import os

class Car:
    def __init__(self):
        fd = os.open("/dev/ttyACM0", os.O_RDWR)
        if fd == -1:
            fd = os.open("/dev/ttyACM1", os.O_RDWR)
        if fd == -1:
            raise NameError("Error opening terminal device")

        attr = termios.tcgetattr(fd)

        attr[1] = attr[1] & ~(termios.OPOST | termios.ONLCR | termios.CBAUD)
        attr[1] |= termios.B9600

        termios.tcsetattr(fd, termios.TCSAFLUSH, attr)
        self.is_on = False
        self.position = 8
        self.file = os.fdopen(fd, "w")
        self.file.write("16\n")

    def turn(self, new_pos):
        self.file.write(str(new_pos)+"\n")
        self.position = new_pos
    def motor(self, on):
        if on:
            self.file.write("17\n")
            self.is_on = True
        else:
            self.file.write("16\n")
            self.is_on = False
        