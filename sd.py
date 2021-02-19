#!/usr/bin/env python3

from trainer import *
from argparse import ArgumentParser

# kördes på ~/data/autocar-round-5 för att få de värden som i arbetet

if __name__ == "__main__":
    parser = ArgumentParser('Finds the standard deviation from a model given the input images.')
    parser.add_argument('-m', '--model', help='The model that is to be evaluated.')
    parser.add_argument('directories', nargs='+', help='The directories in which the images are.')
    parser.add_argument('--linear', action='store_true', help='Needed if the model to be evaluated is linear.')
    args = parser.parse_args()

    model = get_model(linear=args.linear)

    try:
        model = tf.keras.models.load_model(args.model)
    except:
        model = get_model(linear=args.linear)
        model.load_weights(args.model)
    
    imgs, positions = load_imgs(args.directories)

    imgs = pre_proc(imgs, rows_removed=12, break_point=0.5)

    (n, r, c) = np.shape(imgs)
    imgs = np.reshape(imgs, (n, r, c, 1))
    train_sd = np.std(positions)
    if not args.linear:
        positions = to_categorical(positions, num_classes=15)

    preds = []

    for i in range(n):
        raw_prediction = model(imgs[i].reshape((1, 18, 30, 1)))

        _,c = raw_prediction.shape
        if c==1:
            prediction = np.round(raw_prediction)
        else:
            prediction = np.argmax(raw_prediction)
        preds.append(prediction)

    preds = np.array(preds)
    mean = np.mean(preds)
    sd = np.std(preds)
    print(train_sd)
    print(mean)
    print(sd)