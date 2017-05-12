import numpy as np
import sys


def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out.dot(w) + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data.dot(w) + b
    return gen_data


if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    #dim = 10
    N = 500
    # the hidden dimension is randomly chosen from [60, 79] uniformly
    z = np.zeros(50)
    for dim in range(1,61):
       for i in range(50):
          layer_dims = [np.random.randint(60, 80), 100]
          data = gen_data(dim, layer_dims, N)
          z[i] = data.std(axis=0).mean()
       print(z.mean())
