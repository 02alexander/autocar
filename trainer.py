#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import cv2

import test

import argparse

def main():
    parser = argparse.ArgumentParser('trains, creates and evaluates models')
    parser.add_argument('directories', nargs='+', help='the directores in which the training images are.')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--dst')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    rows_removed = 12
    (imgs, positions) = load_imgs(args.directories)
    #(imgs, positions) = load_imgs('/home/alexander/data/autocar-round-5')
    print(np.shape(positions))
    print(np.shape(imgs))
    imgs = pre_proc(imgs, rows_removed=rows_removed, break_point=0.5)

    (n, r, c) = np.shape(imgs)
    imgs = np.reshape(imgs, (n, r, c, 1))

    regs = [round(0.000001*(5**x), 7) for x in range(5)]
    models = [ get_model(reg=reg, linear=args.linear) for reg in regs]
    if models[0].layers[-1].output_shape[1] != 1:
        positions = to_categorical(positions, num_classes=15)
    print(np.shape(positions))
    print(positions)

    fpath = args.dst
    if fpath[-1] != '/':
        fpath += '/'

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath+'r'+str(reg)+"/check",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True) for reg in regs
    ]

    hists = []
    for (model, cb) in zip(iter(models),iter(cbs)):
        hists.append(model.fit(imgs, positions, batch_size=50, epochs=args.epochs, validation_split=0.2, callbacks=[cb]))

    for i in range(len(hists)):
        hist = hists[i]
        plt.ylabel('val_accuracy')
        plt.xlabel('epochs')
        plt.plot(hist.history['val_accuracy'])
    plt.legend(['reg={}'.format(r) for r in regs])
    plt.savefig(fpath+'Figure_1.png')
    
    #test.save_model_weights(models[0], 'wtest')

    
    """(imgs, positions) = load_imgs('/home/alexander/data/autocar-round-5')
    print(np.shape(positions))
    print(np.shape(imgs))
    imgs = pre_proc(imgs, rows_removed=rows_removed, break_point=0.5)

    (n, r, c) = np.shape(imgs)
    imgs = np.reshape(imgs, (n, r, c, 1))
    positions = to_categorical(positions, num_classes=15)

    #test.load_model_weights()

    for model in models:
        eval = model.evaluate(x=imgs, y=positions)
        print(eval)
    """
    #pred_y = models[0](imgs)

    #m = get_model()
    #test.load_model_weights(m, "wtest")

    """for model in models:
        eval = m.evaluate(x=imgs, y=positions)
        print(eval)
    """
    #tf.keras.models.save_model(models[0], "models/lab/test.HD")
    #models[0]

    #for i in range(len(pred_y)):
    #    print(str(np.round(pred_y[i]))+" "+str(np.round(positions[i])))

    #print(np.shape(imgs[0,:]))
    #print(model.predict(imgs[0,:]))
    #keras.models.save_model(model, 'models/conv10_20.HDF5')

def fit_models(models, x, y, prop_val, epochs=1000, batch_size=30):
    n = np.shape(x)[0]
    n_val = round(n*prop_val)
    print("n_val="+str(n_val))
    print("n="+str(n))
    n_train = n-n_val

    trainx = x[:n_train]
    trainy = y[:n_train]
    valx = x[n_train:]
    valy = y[n_train:]

    hists = []
    for model in models:
        hist = model.fit(trainx, trainy, validation_data=(valx, valy), batch_size=batch_size, epochs=epochs)
        hists.append(hist)
    return hists

def get_model(reg=0.0, linear=False):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.7)
    model = Sequential()
    model.add(Conv2D(10, 3,3, activation='sigmoid', input_shape=(18,30,1), kernel_regularizer=l2(reg)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(20, activation='sigmoid', kernel_regularizer=l2(reg)))
    if linear:
        model.add(Dense(1, activation='linear', kernel_regularizer=l2(reg)))
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    else:
        model.add(Dense(15, activation='sigmoid', kernel_regularizer=l2(reg)))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#def get_

def proc_img(img, rows_removed=12, break_point=0.5):
    img = img/255.0
    img = (img>break_point)*1.0
    img = img.reshape(30,30)
    img = img[rows_removed:,:]
    return img.reshape(30-rows_removed, 30)

def pre_proc(data, rows_removed=12, break_point=0.5):
    new_imgs = []
    for img_idx in range(np.shape(data)[0]):
        new_imgs.append(proc_img(data[img_idx], rows_removed=rows_removed, break_point=break_point))
    return np.array(new_imgs)
    #return np.apply_along_axis(f, axis=1, arr=data)

# positions are a integer in the range [0,15)
def load_imgs(directories):
    if type(directories) != list:
        directory = directories
        data = []
        positions = []
        for filename in os.listdir(directory):
            img = cv2.imread(directory+"/"+filename,0)
            data.append(img)
            positions.append(get_servo_pos(filename)-1)
        data = np.array(data)
        return (data, np.array(positions))
    else:
        data = []
        positions = []
        for directory in directories:
            for filename in os.listdir(directory):
                img = cv2.imread(directory+"/"+filename,0)
                data.append(img)
                positions.append(get_servo_pos(filename)-1)
        return (np.array(data), np.array(positions))

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


def preproc_files(srcdir, dstdir):
    for file_name in os.listdir(srcdir):
        org_img = cv2.imread(srcdir+"/"+file_name, 0)
        pimg = proc_img(org_img, rows_removed=12, break_point=0.5)
        org_label = get_servo_pos(file_name)
        random_str = file_name[0:file_name.find("_")]
        cv2.imwrite(dstdir+"/"+"preproc"+random_str+"_"+str(org_label)+".png", pimg*255)


if __name__ == '__main__':
    main()