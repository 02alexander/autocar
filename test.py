#!/usr/bin/env python3

import tensorflow as tf
import pickle
import os

def preproc_files(srcdir, dstdir):
    for file in os.listdir():
        org_img = cv2.imread(srcdir+"/"+file_name)

        org_label = get_servo_pos(file_name)
        random_str = file_name[0:file_name.find("_")]
        cv2.imwrite(dstdir+"/"+"preproc"+random_str+"_"+str(flipped_label)+".png", flipped_img)

def save_model_weights(model, fname):
    pickle.dump(model.get_weights(), open(fname, "wb+"))


def load_model_weights(model, fname):
    weights = pickle.load(open(fname, "rb"))
    model.set_weights(weights)