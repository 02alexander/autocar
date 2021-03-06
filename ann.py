#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import cv2
import os
import termios, sys
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from recorder import Recorder
import argparse
from tensorflow.keras.utils import to_categorical
from trainer import proc_img, get_model

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


def main():
    
    #train()

    parser = argparse.ArgumentParser(description='Controls a lego car autonomously.')
    parser.add_argument('-r', '--record', help='The directory in which the replay is to be stored')
    parser.add_argument('--show', help='Opens a windows that shows what the car sees', action='store_true')
    parser.add_argument('model')
    parser.add_argument('--linear', help='Needed if the model loaded is linear and it\'s weights are stored.', action='store_true')
    args = parser.parse_args()
    rec = None
    if args.record is not None:
        rec = Recorder(args.record)

    fd = os.open("/dev/ttyACM0", os.O_WRONLY|os.O_SYNC)
    cap = cv2.VideoCapture(0)

    try:
        model = tf.keras.models.load_model(args.model)
    except:
        model = get_model(linear=args.linear)
        model.load_weights(args.model) 
    
    first_run = True
    while True:
        input_shape = model.layers[0].input_shape
        rows = None
        if input_shape[1] > 30:
            rows = int(input_shape[1]/30)
        else:
            rows = input_shape[1]
        

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proper = cv2.resize(gray, (30, 30))
        vec = proc_img(proper, rows_removed=30-rows)
        
        if rec is not None:
            rec.store(vec.reshape(22,30)*255)

        if args.show:
            cv2.imshow('wind', vec.reshape(rows,30))


        raw_prediction = None
        if input_shape[1] > 30: # If the model expects a flattened out image. needed for backwards compatibility  
            raw_prediction = model.predict(vec.reshape(1,rows*30))
        else:
            m = np.array(vec)
            raw_prediction = model.predict(m.reshape((1, rows, 30, 1)))



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



if __name__ == "__main__":
    main()