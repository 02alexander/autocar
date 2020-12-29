#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import cv2

def main():
    rows_removed = 12
    (imgs, positions) = load_imgs('/home/alexander/data/autocar-round-5')
    print(np.shape(positions))
    print(np.shape(imgs))
    imgs = pre_proc(imgs, rows_removed=rows_removed, break_point=0.5)
    
    #show_imgs(imgs, positions, positions)
    """
    model = Sequential()

    model.add(Conv2D(2, 2, 2, activation='sigmoid', input_shape=(2,2,1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    x = np.array([
        [1.0,1.0],
        [1.0, 1.0]
    ])
    x = np.reshape(x, (1, 2, 2, 1))
    print(np.shape(x))
    print(model(x))"""

    model = Sequential()
    model.add(Conv2D(10, 3,3, activation='sigmoid', input_shape=(18,30,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(15, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #print(np.shape(imgs[0,:]))
    #print(model.predict(imgs[0,:]))

    if model.layers[-1].output_shape[1] != 1:
        positions = to_categorical(positions, num_classes=15)

    print(np.shape(positions))
    (n, r, c) = np.shape(imgs)
    print(np.shape(imgs))
    print(str(n)+" "+str(r)+" "+str(c))
    imgs = np.reshape(imgs, (n, r, c, 1))
    print(np.shape(imgs))

    train_X = imgs[:3000]
    train_Y = positions[:3000]
    test_X = imgs[3000:]
    test_Y = positions[3000:]

    #model.fit(train_X, train_Y, validation_data=(test_X, test_Y), batch_size=512, epochs=2000)

    model = tf.keras.models.load_model('models/conv10_20.HDF5')

    mse = 0.0
    for i in range(np.shape(test_X)[0]):
        x = np.reshape(test_X[i], (1, 18, 30, 1))
        #print(np.shape(x))
        pred_y = model(x)
        pred_y = np.argmax(pred_y)
        y = np.argmax(test_Y[i], axis=0)
        print(str(pred_y)+" "+str(y))

        if np.square(pred_y-y) > 15:
            display_example(x)

        mse += np.square(pred_y-y)
    mse = mse/np.shape(test_X)[0]
    print(mse)

    #keras.models.save_model(model, 'models/conv10_20.HDF5')


def proc_img(img, rows_removed=8, break_point=0.313):
    img = img/255.0
    img = (img>break_point)*1.0
    img = img.reshape(30,30)
    img = img[rows_removed:,:]
    return img.reshape(30-rows_removed, 30)

def pre_proc(data, rows_removed=8, break_point=0.313):
    new_imgs = []
    for img_idx in range(np.shape(data)[0]):
        new_imgs.append(proc_img(data[img_idx], rows_removed=rows_removed, break_point=break_point))
    return np.array(new_imgs)
    #return np.apply_along_axis(f, axis=1, arr=data)

# positions are a integer in the range [0,15)
def load_imgs(directory):
    data = []
    positions = []
    for filename in os.listdir(directory):
        img = cv2.imread(directory+"/"+filename,0)
        #img = img.reshape(img.size)
        data.append(img)
        positions.append(get_servo_pos(filename)-1)
    data = np.array(data)
    return (data, np.array(positions))

def get_servo_pos(fname):
    underscore_idx = fname.find("_")
    dot_idx = fname.find(".")
    return int(fname[underscore_idx+1:dot_idx])

def show_imgs(X, Y, pred_Y):
    s = X.shape
    r = s[0]
    row_idx = 0
    while row_idx < r:
        print(str(Y[row_idx]) + ", " + str(pred_Y[row_idx]))
        k = display_example(X[row_idx])
        if k == 27:
            break
        if k == 108:
            row_idx -= 2

        row_idx += 1


def display_example(x):
    img = x
    r = None
    c = None
    if len(x.shape) == 1:
        l = x.shape
        img = x.reshape(int(l[0]/30),30)
    else:
        (_,r,c,_) = x.shape
    if r != None and r != 1:
        img = x.reshape(r,30)
    cv2.imshow("example", img)
    k = cv2.waitKey(0)
    cv2.destroyWindow("example")
    return k


if __name__ == '__main__':
    main()