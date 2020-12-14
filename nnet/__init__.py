import os
import numpy as np


class Model:
    def __init__(self, input_shape, n_class,
                 load_weights=False, model_file='nnet.h5'):
        self.__model_path = '%s/ckpt/%s' % (
            os.path.dirname(__file__), model_file)

        self.__nnet = self.make_nnet(input_shape, n_class)

        if load_weights:
            self.nnet.load_weights(self.model_path)

    def make_nnet(self, input_shape, n_class):
        import tensorflow.keras.layers as layers
        from tensorflow.keras import Model

        input_L = layers.Input(shape=(*input_shape, 1))

        conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', activation='relu')(input_L)
        conv1 = layers.Dropout(0.1)(conv1)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', activation='relu')(pool1)
        conv2 = layers.Dropout(0.1)(conv2)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conn = layers.Flatten()(pool2)
        conn = layers.Dense(128, activation='relu')(conn)
        conn = layers.Dropout(0.5)(conn)

        output_L = layers.Dense(n_class, activation='softmax')(conn)

        return Model(input_L, output_L)

    def train(self, x, y, batch_size=32, epochs=20, save_weights=True):
        self.nnet.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.nnet.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )

        if save_weights:
            self.nnet.save_weights(self.model_path)

    def predict(self, x):
        return np.argmax(self.nnet.predict([x])[0])

    @property
    def nnet(self):
        return self.__nnet

    @property
    def model_path(self):
        return self.__model_path
