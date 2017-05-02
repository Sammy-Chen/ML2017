import sys 
from keras.utils.vis_utils import plot_model
from keras.models import load_model


def main():
    emotion_classifier = load_model(sys.argv[1])
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file=sys.argv[2])

if __name__ == '__main__':
    main()
