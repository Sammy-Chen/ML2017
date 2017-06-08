import sys
import numpy as np
import keras
from keras.layers import Embedding, Flatten, Dot, Input
from keras.models import Sequential, Model


def load_data(fname):
    users = []
    movies = []
    ratings = []
    with open(fname) as f:
        #next(f)
        for line in f:
            tid, user, movie, rating = line.strip().split(',')
            users.append(user)
            movies.append(movie)
            ratings.append(rating)
    users = np.array(users, dtype=np.int)
    movies = np.array(movies, dtype=np.int)
    ratings = np.array(ratings, dtype=np.float)
    return users, movies, ratings

np.random.seed(0)

users, movies, ratings = load_data('train.shuf')
n_users = 6040
n_movies = 3952
dim = 120

input_a = Input(shape=[1])
input_b = Input(shape=[1])
emb_a = Embedding(n_users, dim)(input_a)
a = Flatten()(emb_a)
emb_b = Embedding(n_movies, dim)(input_b)
b = Flatten()(emb_b)
out = Dot(axes=1)([a, b])
model = Model([input_a, input_b], out)
model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit([users, movies], ratings, batch_size=128 ,epochs=18, validation_split=0)
model.save('model.h5')
