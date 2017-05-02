import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K

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

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    emotion_classifier = load_model('../my_model.h5')
    np.set_printoptions(precision=2)
    x_val, y_val = read_train_data('../my_val.csv')

    if K.image_data_format() == 'channels_first':
        x_val = x_val.reshape(x_val.shape[0], 1, 48, 48)
    else:
        x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)

    #y_val = keras.utils.to_categorical(y_val, 7)
    predictions = emotion_classifier.predict_classes(x_val)
    conf_mat = confusion_matrix(y_val,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.savefig('confuse.png')

if __name__ == '__main__':
    main()
