import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks
import numpy as np

from constants import *
import generate


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


def train_data_loader(repeat=1):
    with open(generate.TRAIN_OUTPUT_FILEPATH, 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                for _ in range(repeat):
                    yield data
        except EOFError:
            pass


def split_train_data_loader(filename):
    with open(filename, 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass


class Net:
    NUM_EPOCHS = 10000
    STEPS_PER_EPOCH = 1000
    BATCH_SIZE_TRAIN = 24

    def __init__(self):
        self.model = None

    @staticmethod
    def mse(y_true, y_pred):
        """
        MSE loss function for 2D model output (since the built-in
           keras MSE does mean reduction while we want sum reduction)
        """
        return tf.reduce_sum(tf.math.square(y_pred - y_true), axis=[1, 2])

    @staticmethod
    def get_train_dataset():
        def preprocess(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, new_data

        dataset = tf.data.Dataset.from_generator(train_data_loader,
                                                 output_types=tf.float32,
                                                 output_shapes=tf.TensorShape([None, NUM_BINS]))
        # dataset = dataset.cache()

        dataset = dataset.repeat()
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(Net.BATCH_SIZE_TRAIN)
        dataset = dataset.batch(Net.BATCH_SIZE_TRAIN)
        dataset = dataset.prefetch(Net.BATCH_SIZE_TRAIN)

        return dataset

    @staticmethod
    def input_layer():
        return layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')

    @staticmethod
    def get_optimizer():
        lr = 1e-3
        return optimizers.Adam(learning_rate=lr)

    def construct_autoencoder(self):
        net_input = Net.input_layer()
        model = net_input
        
        model = layers.LSTM(256, return_state=True, name='enc1')(model)[2]
        model_repeat = layers.RepeatVector(NUM_FRAMES)(model)
        model = layers.LSTM(256, return_sequences=True, name='dec1')(model_repeat, initial_state=[model, model])
        model = layers.TimeDistributed(layers.Dense(128, activation='relu', name='dec2'))(model)

        net_outputs = model
        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=Net.get_optimizer(), loss=Net.mse)
        
    def construct_encoder(self):
        net_input = Net.input_layer()
        model = net_input
        
        model = layers.LSTM(256, return_state=True, name='enc1')(model)[2]
        
        net_outputs = model
        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=Net.get_optimizer(), loss=Net.mse)

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=True)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def train(self):
        dataset = Net.get_train_dataset()

        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-epoch{epoch:05d}-loss{loss:.2f}.hdf5'),
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )

        tensorboard_callback = callbacks.TensorBoard(
            log_dir='logs',
            write_graph=False,
            profile_batch=10
        )

        self.model.fit(x=dataset,
                       epochs=Net.NUM_EPOCHS,
                       steps_per_epoch=Net.STEPS_PER_EPOCH,
                       callbacks=[checkpoint_callback],
                       verbose=1)


def main():
    net = Net()
    net.construct_autoencoder()
    #tf.keras.utils.plot_model(net.model, to_file='model_train.png', show_shapes=True)
    #net.load_weights('checkpoints/weights.hdf5')
    net.train()


if __name__ == '__main__':
    main()
