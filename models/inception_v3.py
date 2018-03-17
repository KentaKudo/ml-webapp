from keras.applications.inception_v3 import InceptionV3 as Iv3
from keras.layers import Dense
from keras.models import Model

def InceptionV3():
  inception_v3 = Iv3(weights=None, input_shape=(225,225,3))
  x = Dense(1, activation='linear')(inception_v3.outputs[0])
  model = Model(inception_v3.inputs[0], x)
  
  return model
