import sys
import numpy as np
from PIL import Image
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('TKAgg')

letters = 'ABCDEFGHIJ'
images = np.zeros((100, 64 * 64))
for i in range(10):
    for j in range(10):
        img = Image.open('data/' + letters[i] + '0' + str(j) + '.bmp')
        data = np.asarray(img).flatten()
        images[i * 10 + j] = data

avgFace = images.mean(axis=0)
"""
plt.imshow(avgFace.reshape(64,64), cmap='gray')
plt.title('average-face')
plt.savefig('fig1')
"""
fig3 = plt.figure()
plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
for i in range(10 * 10):
    ax = fig3.add_subplot(10, 10, i + 1)
    ax.imshow(images[i].reshape(64 ,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig('fig3')

images = images - avgFace

u, s, v = np.linalg.svd(images.T)
"""
fig = plt.figure()
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(u[i].reshape(64 ,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()

plt.savefig('fig2')
"""
k = int(sys.argv[1])
reImages = (u[:,:k].dot(np.diag(s[:k])).dot(v[:k,:])).T
print(reImages.shape)
print(np.sqrt(np.mean((images - reImages) ** 2)) / 256)
reImages += avgFace
fig4 = plt.figure()
for i in range(10 * 10):
    ax = fig4.add_subplot(10, 10, i + 1)
    ax.imshow(reImages[i].reshape(64 ,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig('fig4')
