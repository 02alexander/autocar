#!/usr/bin/env python3

from trainer import *
import matplotlib.pyplot as plt
import tensorflow as tf

epochs = [1200, 600, 400, 300]
rows_removed = 12



def experiment_classification():
    for i in range(4):
        directories = ['data/manual_round'+str(k) for k in range(1, i+2)]
        imgs, positions = load_imgs(directories)

        imgs = pre_proc(imgs, rows_removed=rows_removed, break_point=0.5)
        (n, r, c) = np.shape(imgs)
        imgs = np.reshape(imgs, (n, r, c, 1))

        regs = [round(0.000001*(5**x), 7) for x in range(5)]
        models = [ get_model(reg=reg) for reg in regs]
        positions = to_categorical(positions, num_classes=15)
        cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='experiments/class'+str(i+1)+'/r'+str(reg)+"/check",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True) for reg in regs
        ]

        hists = []
        for (model, cb) in zip(iter(models),iter(cbs)):
            hists.append(model.fit(imgs, positions, batch_size=50, epochs=epochs[i], validation_split=0.2, callbacks=[cb]))

        for c in range(len(hists)):
            hist = hists[c]
            plt.ylabel('val_accuracy')
            plt.xlabel('epochs')
            plt.plot(hist.history['val_accuracy'])
        plt.legend(['reg={}'.format(r) for r in regs])
        plt.savefig('experiments/class'+str(i+1)+'/Figure_1.png')
        plt.clf()

def experiment_linear():
    for i in range(1):
        directories = ['data/manual_round'+str(k) for k in range(1, i+2)]
        imgs, positions = load_imgs(directories)

        imgs = pre_proc(imgs, rows_removed=rows_removed, break_point=0.5)
        (n, r, c) = np.shape(imgs)
        imgs = np.reshape(imgs, (n, r, c, 1))

        regs = [round(0.000001*(5**x), 7) for x in range(5)]
        models = [ get_model(reg=reg, linear=True) for reg in regs]

        cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='experiments/linear'+str(i+1)+'/r'+str(reg)+"/check",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True) for reg in regs
        ]

        hists = []
        for (model, cb) in zip(iter(models),iter(cbs)):
            hists.append(model.fit(imgs, positions, batch_size=50, epochs=epochs[i], validation_split=0.2, callbacks=[cb]))

        for c in range(len(hists)):
            hist = hists[c]
            plt.ylabel('loss')
            plt.xlabel('epochs')
            plt.plot(hist.history['val_loss'])
        plt.legend(['reg={}'.format(r) for r in regs])
        plt.savefig('experiments/linear'+str(i+1)+'/Figure_1.png')
        plt.clf()


if __name__ == "__main__":
    #experiment_classification()
    experiment_linear()