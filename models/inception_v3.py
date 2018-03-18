from keras.applications.inception_v3 import InceptionV3 as Iv3
from keras.layers import Dense
from keras.models import Model

image_size = (225,225)

def InceptionV3(num_classes=1000):
    return Iv3(weights=None, input_shape=image_size+(3,), classes=num_classes)
