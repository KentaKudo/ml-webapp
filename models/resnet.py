def ResNet(num_classes):
    from keras.models import Model
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization, Add, Activation, GlobalAveragePooling2D
    from keras.regularizers import l2

    def resblock(filters, kernel_size=(3, 3), increase=False):
        strides = (2, 2) if increase else (1, 1)
        def _res_block(x):
            x_ = Conv2D(filters, kernel_size,
                strides=strides,
                padding='same',
                kernel_regularizer=l2(weight_decay),
                activation='relu')(x)
            x_ = BatchNormalization()(x_)
            x_ = Conv2D(filters, kernel_size,
                strides=(1, 1),
                padding='same',
                kernel_regularizer=l2(weight_decay),
                activation='relu')(x_)
            if increase:
              x = Conv2D(filters, (1, 1),
                  strides=(2, 2),
                  padding='same',
                  kernel_regularizer=l2(weight_decay),
                  activation='relu')(x)
            x = Add()([x_, x])
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
        return _res_block

    weight_decay = 1e-4
    inputs = Input(shape=image_size+(3,))

    # 225 * 225 * 3
    x = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 112 * 112 * 64 → 112 * 112 * 64
    x = resblock(64, increase=True)(x)
    x = resblock(64)(x)
    x = resblock(64)(x)

    # 112 * 112 * 64 → 66 * 66 * 64
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 66 * 66 * 64 → 66 * 66 * 128
    x = resblock(128, increase=True)(x)
    x = resblock(128)(x)
    x = resblock(128)(x)
    x = resblock(128)(x)

    # 66 * 66 * 128 → 33 * 33 * 128
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 33 * 33 * 128 → 33 * 33 * 256
    x = resblock(256, increase=True)(x)
    x = resblock(256)(x)
    x = resblock(256)(x)
    x = resblock(256)(x)
    x = resblock(256)(x)
    x = resblock(256)(x)

    # 33 * 33 * 256 → 17 * 17 * 256
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 17 * 17 * 256 → 17 * 17 * 512
    x = resblock(512, increase=True)(x)
    x = resblock(512)(x)
    x = resblock(512)(x)

    # flatten
    x = GlobalAveragePooling2D()(x)
    #  flattened → 1000
    x = Dense(1000, activation='relu', kernel_initializer='he_normal')(x)
    #  1000 → num_classes
    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=y)
    return model
