import sys
import numpy as np
from PIL import Image

images = np.zeros((481, 480 * 512))
for i in range(481):
   img = Image.open('data/hand/hand.seq' + str(i+1) + '.png')
   data = np.asarray(img).flatten()
   images[i] = data

images = images - images.mean(axis=0)
index = np.array(list(range(481)))
np.random.shuffle(index)

s_sum = 0
k = 20
for i in range(5):
   u, s, v = np.linalg.svd(images[index[k*i:k*(i+1)]].T, full_matrices=0)
   print(s)

