import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

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

x_train, y_train = read_train_data(sys.argv[1])

batch_size = 128
num_classes = 7
epochs = 20

y_train = keras.utils.to_categorical(y_train, num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
model.add(Dense(2048, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          shuffle=True)
model.save('nn_model.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('training procedure')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('nnfig')
