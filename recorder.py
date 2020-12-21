#!/usr/bin/env python3

import cv2
import os
import sys

class Recorder:
    # directory is a string
    def __init__(self, directory):
        self.directory = directory
        if self.directory[-1] != '/':
            self.directory += '/'
        self.cur_iter = self.get_cur_iter()
    def get_cur_iter(self):
        largest_iter = 0
        for filename in os.listdir(self.directory):
            digit = get_file_digit(filename)
            if digit >= largest_iter:
                largest_iter = digit
        return largest_iter+1
    def store(self, image): # cv2 image
        filename = self.directory
        filename += str(self.cur_iter)+".png" 
        cv2.imwrite(filename, image)
        self.cur_iter += 1
    def replay(self):
        filenames = os.listdir(self.directory)
        filenames.sort(key=get_file_digit)
        i = 0
        while i < len(filenames):
            img = cv2.imread(self.directory+filenames[i])
            cv2.imshow('recorder-replay', img)
            print(filenames[i])
            k = cv2.waitKey(0)
            print(k)
            if k == 108: # l
                i = i-2
            if k == 27: # escape
                break
            i = i+1
        cv2.destroyWindow('recorder-replay')


def get_file_digit(filename):
    dot_idx = filename.find('.')
    return int(filename[0:dot_idx])


def main():
    print(sys.argv[1])
    rec = Recorder(sys.argv[1])
    rec.replay()

if __name__ == '__main__':
    main()