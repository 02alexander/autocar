#!/usr/bin/env python3

import numpy as np
import matplotlib
import serial
import tensorflow as tf
import cv2
import os
import termios, sys

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

def train(x, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50,activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(50,activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1)
    ])
    model.compile('sgd',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
                  
    model.fit(x, y, batch_size=100, epochs=1000)
    return model

def feature_scaling(row):
    min = np.min(row)
    max = np.max(row)
    return (row-min)/(max-min)

def pre_proc(data):
    return np.apply_along_axis(feature_scaling, axis=1,arr=data)
    

if __name__ == "__main__":

    fd = os.open("/dev/ttyACM0", os.O_WRONLY|os.O_SYNC)
    os.write(fd, b"14")

    cap = cv2.VideoCapture(0)

    model = tf.keras.models.load_model("model.HDF5")
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proper = cv2.resize(gray, (30, 30))
        vec = proper.reshape((1,proper.size))

        prediction = np.round(model.predict(vec), 0)+1
        print(prediction)
        os.write(fd, bytes(str(prediction), 'ASCII'))
        cv2.waitKey(100)

    """(x,y) = load_imgs("/home/alexander/data/autocar-round-3")
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

