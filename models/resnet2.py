from . import image_size

stack_n = 5

# https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py
def ResNet2(num_classes):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense
    from keras.regularizers import l2

    def resblock(filters, increase=False):
        stride = (2, 2) if increase else (1, 1)

        def _resblock(x):
            o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
            conv_1 = Conv2D(filters, kernel_size=(3,3),
                            strides=stride,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(o1)
            o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
            conv_2 = Conv2D(filters, kernel_size=(3,3),
                            strides=(1,1),
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(o2)
            if increase:
                projection = Conv2D(filters, kernel_size=(1,1),
                                    strides=(2,2),
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=l2(weight_decay))(o1)
                block = Add()([conv_2, projection])
            else:
                block = Add()([conv_2, x])
            return block
        return _resblock

    inputs = Input(shape=image_size+(3,))
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(inputs)
    
    for _ in range(stack_n):
        x = resblock(64, False)(x)
    x = resblock(128, True)(x)
    for _ in range(1, stack_n):
        x = resblock(128, False)(x)
    x = resblock(256, True)(x)
    for _ in range(1, stack_n):
        x = resblock(256, False)(x)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    y = Dense(num_classes, activation='softmax', kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=inputs, outputs=y)
    return model
