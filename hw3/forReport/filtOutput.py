import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
#from utils import *
#from marcos import *
import numpy as np

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
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ['conv2d_1']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    x_val, y_val = read_train_data('../my_val.csv')
    if K.image_data_format() == 'channels_first':
        x_val = x_val.reshape(x_val.shape[0], 1, 1, 48, 48)
    else:
        x_val = x_val.reshape(x_val.shape[0], 1, 48, 48, 1)

    choose_id = 0
    photo = x_val[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure()
        #nb_filter = im[0].shape[3]
        nb_filter = 32
        print('what: ',nb_filter)
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.savefig(os.path.join('img/','layer{}'.format(cnt)))

if __name__ == '__main__':
    main()
