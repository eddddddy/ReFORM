import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks
import numpy as np

from constants import *
import generate

# tf.compat.v1.disable_eager_execution()

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


def ConvBN(*args, **kwargs):
    def _ConvBN(model, training=True):
        model = layers.Conv2D(*args, **{k: v for k, v in kwargs.items() if k != 'name'}, name=f'conv_{kwargs["name"]}')(model)
        model = layers.BatchNormalization(axis=-1, trainable=training, name=f'bn_{kwargs["name"]}')(model, training=training)
        model = layers.ReLU()(model)
        return model

    return _ConvBN


def ConvTransposeBN(*args, **kwargs):
    def _ConvTransposeBN(model, training=True):
        model = layers.Conv2DTranspose(*args, **{k: v for k, v in kwargs.items() if k != 'name'}, name=f'conv_transpose_{kwargs["name"]}')(model)
        model = layers.BatchNormalization(axis=-1, trainable=training, name=f'bn_{kwargs["name"]}')(model, training=training)
        model = layers.ReLU()(model)
        return model

    return _ConvTransposeBN


class Net:
    NUM_EPOCHS = 10000
    STEPS_PER_EPOCH = 512 * 8
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_CLUSTER = 32

    def __init__(self):
        self.model = None

    @staticmethod
    def mse(y_true, y_pred):
        """
        MSE loss function for 2D model output (since the built-in
           keras MSE doesn't behave the way we want it to)
        """
        return tf.reduce_sum(tf.math.square(y_pred - y_true), axis=[1, 2])

    @staticmethod
    def get_train_dataset():
        def slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - REVAMP_NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + REVAMP_NUM_FRAMES]
            return new_data, new_data

        dataset = tf.data.Dataset.from_generator(train_data_loader,
                                                 output_types=tf.float64,
                                                 output_shapes=tf.TensorShape([None, NUM_BINS]))
        dataset = dataset.cache()

        dataset = dataset.repeat()
        dataset = dataset.map(slice_n_dice, num_parallel_calls=8)
        dataset = dataset.shuffle(Net.BATCH_SIZE_TRAIN * 2)
        dataset = dataset.batch(Net.BATCH_SIZE_TRAIN)
        dataset = dataset.prefetch(Net.BATCH_SIZE_TRAIN * 2)

        return dataset

    @staticmethod
    def input_layer():
        return layers.Input(shape=(REVAMP_NUM_FRAMES, NUM_BINS), name='input')

    @staticmethod
    def encoder_conv(model, filters, kernel_size, strides, tag, padding='same', training=True):
        model = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, name=f'conv_encoder_{tag}')(model)
        model = layers.BatchNormalization(axis=-1, trainable=training, name=f'bn_encoder_{tag}')(model, training=training)
        model = layers.ReLU()(model)
        return model

    @staticmethod
    def decoder_conv(model, filters, kernel_size, strides, tag, padding='same', training=True):
        model = layers.UpSampling1D(strides)(model)
        model = layers.Conv1D(filters, kernel_size, padding=padding, name=f'conv_decoder_{tag}')(model)
        model = layers.BatchNormalization(axis=-1, trainable=training, name=f'bn_decoder_{tag}')(model, training=training)
        model = layers.ReLU()(model)
        return model

    @staticmethod
    def res_block(model, in_filters, filters, tag, training=True):
        out = layers.Conv1D(filters, 3, padding='same', name=f'res_conv_{tag}_1')(model)
        out = layers.BatchNormalization(axis=-1, trainable=training, name=f'res_bn_{tag}_1')(out, training=training)
        out = layers.ReLU()(out)
        out = layers.Conv1D(filters, 3, padding='same', name=f'res_conv_{tag}_2')(out)
        out = layers.BatchNormalization(axis=-1, trainable=training, name=f'res_bn_{tag}_2')(out, training=training)
        out = layers.ReLU()(out)
        out = layers.Conv1D(filters, 3, padding='same', name=f'res_conv_{tag}_3')(out)
        out = layers.BatchNormalization(axis=-1, trainable=training, name=f'res_bn_{tag}_3')(out, training=training)
        out = layers.ReLU()(out)

        if in_filters == filters:
            shortcut = model
        else:
            shortcut = layers.Conv1D(filters, 1, padding='same', name=f'res_shortcut_conv_{tag}')(model)
            shortcut = layers.BatchNormalization(axis=-1, trainable=training, name=f'res_shortcut_bn_{tag}')(shortcut, training=training)

        out = layers.Add()([out, shortcut])
        out = layers.ReLU()(out)

        return out

    @staticmethod
    def get_optimizer():
        lr = 1e-3
        return optimizers.Adam(learning_rate=lr)

    def construct_for_training(self):
        net_input = Net.input_layer()
        model = net_input

        model = Net.encoder_conv(model, 96, 7, 2, 'conv_1')
        model = layers.MaxPool1D(pool_size=2)(model)
        model = Net.encoder_conv(model, 96, 3, 1, 'conv_2')
        model = layers.MaxPool1D(pool_size=2)(model)
        model = Net.encoder_conv(model, 128, 3, 1, 'conv_3')
        model = layers.MaxPool1D(pool_size=2)(model)
        model = Net.encoder_conv(model, 128, 3, 1, 'conv_4')
        model = layers.MaxPool1D(pool_size=2)(model)
        model = Net.encoder_conv(model, 192, 3, 1, 'conv_5')
        model = layers.MaxPool1D(pool_size=2)(model)
        model = Net.encoder_conv(model, 192, 3, 1, 'conv_6', padding='valid')
        model = Net.encoder_conv(model, 256, 2, 1, 'conv_7', padding='valid')

        model = layers.Reshape((256,))(model)
        model = layers.Dense(256, activation='relu', name='dense_1')(model)
        model = layers.Dense(256, activation='relu', name='dense_2')(model)
        model = layers.Reshape((1, 256))(model)

        model = Net.decoder_conv(model, 192, 3, 2, 'dconv_1')
        model = Net.decoder_conv(model, 192, 3, 2, 'dconv_2')
        model = Net.decoder_conv(model, 128, 3, 2, 'dconv_3')
        model = Net.decoder_conv(model, 128, 3, 2, 'dconv_4')
        model = Net.decoder_conv(model, 96, 3, 2, 'dconv_5')
        model = Net.decoder_conv(model, 96, 3, 2, 'dconv_6')
        model = Net.decoder_conv(model, 128, 9, 4, 'dconv_7')

        model = layers.Conv1D(128, 7, strides=1, activation='sigmoid', padding='same', name='dconv_8')(model)

        net_outputs = model

        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=Net.get_optimizer(), loss=Net.mse)

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=True)

    def predict(self, x):
        return self.model.predict(x)

    def train(self):
        dataset = Net.get_train_dataset()

        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'revamp_relu_weights-epoch{epoch:05d}-loss{loss:.2f}.hdf5'),
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
    net.construct_for_training()
    tf.keras.utils.plot_model(net.model, to_file='model_revamp_train.png', show_shapes=True)
    #net.load_weights('checkpoints/revamp-loss197.03.hdf5')
    net.train()


if __name__ == '__main__':
    main()
