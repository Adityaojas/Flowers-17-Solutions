from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K


class MiniVGGNet:
    @staticmethod

    def build(width, height, depth, classes):
        
        model = Sequential()
        input_shape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chanDim = 1
        
        model.add(Conv2D(32, (3,3), padding ='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
        
        
        
        
        
        
        
        