import os
import argparse
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from vis.utils import utils
#from vis.visualization import visualize_saliency

def read_train_data(fname):
    x_train = []
    y_train = []
    with open(fname) as f:
        # skip the first row
        next(f)
        for line in f:
            label, features = line.strip().split(',')
            x_train.append(features.split())
            y_train.append(label)
    x_train = np.array(x_train, dtype=np.float)
    x_train = (x_train - x_train.mean()) / x_train.std()
    y_train = np.array(y_train, dtype=np.int)
    return x_train, y_train

def main():
    emotion_classifier = load_model('../my_model.h5')
    x_val, y_val = read_train_data('../my_val.csv')
    if K.image_data_format() == 'channels_first':
        x_val = x_val.reshape(x_val.shape[0], 1, 1, 48, 48)
    else:
        x_val = x_val.reshape(x_val.shape[0], 1, 48, 48, 1)

    input_img = emotion_classifier.input
    img_ids = [0,2,3,8,13,15,24]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(x_val[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        fn = K.function([input_img, K.learning_phase()], [grads])

         
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        heatmap = fn([x_val[idx], 0])[0]
        heatmap = heatmap.reshape(48, 48) 
        heatmap = np.abs(heatmap) 
        heatmap = heatmap / heatmap.max()


        thres = 0.075
        see = x_val[idx].reshape(48, 48)
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('img/original' + str(y_val[idx]) + '_' + str(pred) + '.png', dpi=100)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('img/salien' + str(y_val[idx]) + '_' + str(pred) + '.png', dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('img/see' + str(y_val[idx]) + '_' + str(pred) + '.png', dpi=100)

if __name__ == "__main__":
    main()
