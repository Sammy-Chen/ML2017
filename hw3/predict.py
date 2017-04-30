import sys
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K

def read_test_data(fname):
    x_test = []
    ids = []
    with open(fname) as f:
        # skip the first row
        next(f)
        for line in f:
            i, features = line.strip().split(',')
            x_test.append(features.split())
            ids.append(i)
    x_test = np.array(x_test, dtype=np.float)
    x_test = (x_test - x_test.mean()) / x_test.std()
    return ids, x_test

ids, x_test = read_test_data(sys.argv[1])
batch_size = 128
img_rows, img_cols = 48, 48

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#x_test /= 255

model = load_model(sys.argv[2])
ans = model.predict_classes(x_test, batch_size = batch_size, verbose = 0)

with open(sys.argv[3], 'w') as f:
    f.write('id,label\n')
    for i, label in zip(ids, ans):
        f.write('{0},{1}\n'.format(i, label))
