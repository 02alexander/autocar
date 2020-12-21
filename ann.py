#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2
import os
import termios, sys
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from recorder import Recorder
import argparse

def load_imgs(directory):
    data = []
    positions = []
    for filename in os.listdir(directory):
        img = cv2.imread(directory+"/"+filename,0)
        img = img.reshape(img.size)
        data.append(img)
        positions.append(get_servo_pos(filename)-1)
    data = np.array(data)
    return (data, np.array(positions))

def get_servo_pos(fname):
    underscore_idx = fname.find("_")
    dot_idx = fname.find(".")
    return int(fname[underscore_idx+1:dot_idx])

def train_single_output(x, y, x_test=None, y_test=None, epochs=300, reg=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20,activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(reg)),
        tf.keras.layers.Dense(20,activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(reg)),
        tf.keras.layers.Dense(1,activation=tf.nn.relu)
    ])
    model.compile(tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mean_absolute_error'])
    
    if x_test is not None and y_test is not None:
        return (model, model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test)))
    else:
        return (model, model.fit(x, y, epochs=epochs))

def train_models(x, y, epochs, regs):
    models = []
    for reg in regs:
        (model,_) = train_single_output(x, y, epochs=epochs, reg=reg)
        models.append(model)
    return models

def show_imgs(X, Y, pred_Y):
    (r,c) = X.shape
    for row_idx in range(r):
        print(str(Y[row_idx]) + ", " + str(pred_Y[row_idx]))
        display_example(X[row_idx])

def display_example(x):
    img = x
    r = None
    c = None
    if len(x.shape) == 1:
        l = x.shape
        img = x.reshape(int(l[0]/30),30)
    else:
        (r,c) = x.shape
    if r != None and r != 1:
        img = x.reshape(30,30)
    cv2.imshow("example", img)
    cv2.waitKey(0)
    cv2.destroyWindow("example")

def train_15_outputs(model, x, y, x_test=None, y_test=None, reg=0.0, epochs=300):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(reg)),
        tf.keras.layers.Dense(50, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(reg)),
        tf.keras.layers.Dense(15, activation=tf.nn.sigmoid)
    ])
    model.compile('adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if x_test is not None and y_test is not None:
        return model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test))
    else:
        return model.fit(x, y, epochs=epochs)

def train_bounded_output(x, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(120, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(50, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.relu)
    ])
    model.compile('sgd',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    model.fit(x, y, batch_size=700, epochs=300)
    return model

def feature_scaling(row):
    min = np.min(row)
    max = np.max(row)
    return (row-min)/(max-min)

def proc_row(row):
    row = row/255.0
    row = (row>0.313)*1.0
    img = row.reshape(30,30)
    img = img[8:,:]
    return img.reshape(22*30)

def pre_proc(data):
    return np.apply_along_axis(proc_row, axis=1, arr=data)

# takes every image in srcdir then flips it and stores the flipped image as 
# flipped<original random str>_<flipped label>.png in dstdir
# for example FA54HG_1.png becomes flipped_FA54HG_15.png
def create_flipped_dataset(srcdir, dstdir):
    for file_name in os.listdir(srcdir):
        org_img = cv2.imread(srcdir+"/"+file_name)
        flipped_img = cv2.flip(org_img, 1)
        org_label = get_servo_pos(file_name)
        flipped_label = 16-org_label
        random_str = file_name[0:file_name.find("_")]
        cv2.imwrite(dstdir+"/"+"flipped"+random_str+"_"+str(flipped_label)+".png", flipped_img)

"""
def train():
    (x,y) = load_imgs("/home/alexander/data/autocar-round-5,6")
    print(np.shape(x))

    model = tf.keras.Sequential([
        Conv2D(10, 3, 3, activation=tf.nn.sigmoid),
        Flatten(),
        Dense(50, activation=tf.nn.sigmoid),
        Dense(15, activation=tf.nn.sigmoid)
    ])
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
"""

def main():
    
    #train()

    parser = argparse.ArgumentParser(description='Controls a lego car autonomously.')
    parser.add_argument('-r', '--record', help='The directory in which the replay is to be stored')
    parser.add_argument('--show', help='Opens a windows that shows what the car sees', action='store_true')
    parser.add_argument('model')
    args = parser.parse_args()
    rec = None
    if args.record is not None:
        rec = Recorder(args.record)

    fd = os.open("/dev/ttyACM0", os.O_WRONLY|os.O_SYNC)
    cap = cv2.VideoCapture(0)

    model = tf.keras.models.load_model(args.model)
    first_run = True
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proper = cv2.resize(gray, (30, 30))
        vec = proper.reshape((1,proper.size))
        vec = proc_row(vec).reshape(1,22*30)
        
        if rec is not None:
            rec.store(vec.reshape(22,30)*255)

        if args.show:
            cv2.imshow('wind', vec.reshape(22,30))

        raw_prediction = model.predict(vec)
        prediction = None
        (_,c) = raw_prediction.shape
        if c == 1:
            prediction = np.round(raw_prediction, 0)
        else:
            prediction = np.argmax(raw_prediction,axis=1)[0]
        prediction += 1
        print(prediction)
        if first_run:
            os.write(fd, bytes(str(17)+"\x0a\x0d", 'ASCII'))
            first_run=False 
        os.write(fd, bytes(str(prediction)+"\x0a\x0d", 'ASCII'))
        cv2.waitKey(30)

    if args.show:
        cv2.destroyAllWindows()

    """
    (x,y) = load_imgs("/home/alexander/data/autocar-round-3")
    x = pre_proc(x)
    y = y - 1
    print(y)

    x_train = x[0:400]
    y_train = y[0:400]
    x_test = x[400:]
    y_test = y[400:]
    train(x_train, y_train)
    print(x_train.shape)
    print(x_test.shape)
    print(x[0].reshape(1,900).shape)

    model = train(x_train, y_train)


    print(x[0].reshape(1,900).shape)
    print(np.round(model.predict(x[0:10])).transpose())
    print(y[0:10])
    model.evaluate(x_test,y_test)"""



if __name__ == "__main__":
    main()