import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
#from utils import *
#from marcos import *
import numpy as np

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    eta = 0.5
    mytarget = 0.0
    for i in range(num_step):
        target, grads = iter_func([input_image_data, 0])
        input_image_data += grads * eta 
        mytarget = target

    return [input_image_data.reshape(48,48), mytarget] 

def main():
    emotion_classifier = load_model('../my_model.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ['conv2d_1']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    nb_filter = 32
    NUM_STEPS = 100
    RECORD_FREQ = 100
    for cnt, c in enumerate(collect_layers):
        #filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        filter_imgs = []
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])

            ###
            "You need to implement it."
            filter_imgs.append(grad_ascent(NUM_STEPS, input_img_data, iterate))
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure()
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/8, 8, i+1)
                ax.imshow(filter_imgs[i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], (it + 1)*RECORD_FREQ))
            fig.savefig('./filter.png')

if __name__ == "__main__":
    main()
