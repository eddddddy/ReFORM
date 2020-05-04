import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks
import numpy as np

from constants import *
import generate

tf.compat.v1.disable_eager_execution()

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
                data = pickle.load(f)
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
    STEPS_PER_EPOCH = 128
    BATCH_SIZE_PRETRAIN = 256
    BATCH_SIZE_CLUSTER = 32

    def __init__(self):
        self.model = None

    @staticmethod
    def spreader(batch_size):
        """
        Custom loss function that ignores y_true and calculates
        the mean squared error between y_pred and the mean of every
        vector in y_pred (i.e. it calculates how spread out the
        points of y_pred are)
        """

        def spread(y_true, y_pred):
            y_avg = tf.tile(tf.reduce_mean(y_pred, axis=0, keepdims=True), [batch_size, 1])
            return tf.reduce_sum(tf.math.square(y_pred - y_avg), axis=1)

        return spread

    @staticmethod
    def mse(y_true, y_pred):
        """
        MSE loss function for 2D model output (since the built-in
           keras MSE doesn't behave the way we want it to)
        """
        return tf.reduce_sum(tf.math.square(y_pred - y_true), axis=[1, 2])

    @staticmethod
    def get_pretrain_dataset():
        def __slice_n_dice(data, length):
            data_offset = tf.random.uniform([], 0, length - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, {'conv1_aux': new_data, 'conv2_aux': new_data, 'conv3_aux': new_data, 'conv4_aux': new_data, 'conv5_aux': new_data, 'decoder': new_data}

        data = list(train_data_loader())
        lengths = np.array([len(d) for d in data])
        for i in range(len(data)):
            data[i] = np.pad(data[i], [(0, 35000 - len(data[i])), (0, 0)])
        data = np.array(data)

        data = tf.convert_to_tensor(data[100:])
        lengths = tf.convert_to_tensor(lengths[100:])

        '''data1 = tf.convert_to_tensor(data[:100])
        lengths1 = tf.convert_to_tensor(lengths[:100])
        data2 = tf.convert_to_tensor(data[100:])
        lengths2 = tf.convert_to_tensor(lengths[100:])'''

        '''dataset1 = tf.data.Dataset.range(8).interleave(
            lambda x: tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(data1[x::8]),
                tf.data.Dataset.from_tensor_slices(lengths1[x::8])
            )),
            cycle_length=8,
            num_parallel_calls=8
        )
        dataset2 = tf.data.Dataset.range(8).interleave(
            lambda x: tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(data2[x::8]),
                tf.data.Dataset.from_tensor_slices(lengths2[x::8])
            )),
            cycle_length=8,
            num_parallel_calls=8
        )'''
        dataset = tf.data.Dataset.range(4).interleave(
            lambda x: tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(data[x::8]),
                tf.data.Dataset.from_tensor_slices(lengths[x::8])
            )),
            cycle_length=4,
            num_parallel_calls=4
        )
        #dataset = dataset1.concatenate(dataset2)
        dataset = dataset.repeat()
        dataset = dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(Net.BATCH_SIZE_PRETRAIN)
        dataset = dataset.prefetch(Net.BATCH_SIZE_PRETRAIN * 4)

        return dataset
    
    @staticmethod
    def get_pretrain_dataset_new():
        def slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, new_data
            
        def dataset_from_filename(filename):
            return tf.data.Dataset.from_generator(
                split_train_data_loader,
                output_types=tf.float64,
                output_shapes=tf.TensorShape([None, NUM_BINS]),
                args=(filename,)
            )
        
        train_files = [generate.get_nth_train_filepath(i) for i in range(generate.NUM_THREADS)]
        dataset = tf.data.Dataset.from_tensor_slices(train_files)
        dataset = dataset.interleave(dataset_from_filename, cycle_length=generate.NUM_THREADS)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(Net.BATCH_SIZE_PRETRAIN * 4)
        dataset = dataset.map(slice_n_dice)
        dataset = dataset.batch(Net.BATCH_SIZE_PRETRAIN)
        dataset = dataset.prefetch(Net.BATCH_SIZE_PRETRAIN * 4)
        
        return dataset

    @staticmethod
    def input_layer():
        return layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')

    def construct_for_pretraining(self):
        net_input = Net.input_layer()
        model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(net_input)

        conv1 = ConvBN(64, (11, 9), strides=(5, 4), padding='same', name=1)(model)
        conv1_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux1_1')(conv1)
        conv1_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv1_aux')(conv1_aux)

        conv2 = ConvBN(128, (9, 5), strides=(4, 2), padding='same', name=2)(conv1)
        conv2_aux = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='aux2_1')(conv2)
        conv2_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux2_2')(conv2_aux)
        conv2_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv2_aux')(conv2_aux)

        conv3 = ConvBN(256, (9, 5), strides=(4, 2), padding='same', name=3)(conv2)
        conv3_aux = ConvTransposeBN(128, (5, 3), strides=(4, 2), padding='same', name='aux3_1')(conv3)
        conv3_aux = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='aux3_2')(conv3_aux)
        conv3_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux3_3')(conv3_aux)
        conv3_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv3_aux')(conv3_aux)

        conv4 = ConvBN(384, (5, 5), strides=(2, 2), padding='same', name=4)(conv3)
        conv4_aux = ConvTransposeBN(256, (3, 3), strides=(2, 2), padding='same', name='aux4_1')(conv4)
        conv4_aux = ConvTransposeBN(128, (5, 3), strides=(4, 2), padding='same', name='aux4_2')(conv4_aux)
        conv4_aux = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='aux4_3')(conv4_aux)
        conv4_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux4_4')(conv4_aux)
        conv4_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv4_aux')(conv4_aux)

        conv5 = ConvBN(512, (5, 5), strides=(2, 2), padding='same', name=5)(conv4)
        conv5_aux = ConvTransposeBN(384, (3, 3), strides=(2, 2), padding='same', name='aux5_1')(conv5)
        conv5_aux = ConvTransposeBN(256, (3, 3), strides=(2, 2), padding='same', name='aux5_2')(conv5_aux)
        conv5_aux = ConvTransposeBN(128, (5, 3), strides=(4, 2), padding='same', name='aux5_3')(conv5_aux)
        conv5_aux = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='aux5_4')(conv5_aux)
        conv5_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux5_5')(conv5_aux)
        conv5_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv5_aux')(conv5_aux)

        conv6 = ConvBN(768, (5, 5), strides=(2, 2), padding='same', name=6)(conv5)
        decoder = ConvTransposeBN(512, (3, 3), strides=(2, 2), padding='same', name='decoder_1')(conv6)
        decoder = ConvTransposeBN(384, (3, 3), strides=(2, 2), padding='same', name='decoder_2')(decoder)
        decoder = ConvTransposeBN(256, (3, 3), strides=(2, 2), padding='same', name='decoder_3')(decoder)
        decoder = ConvTransposeBN(128, (5, 3), strides=(4, 2), padding='same', name='decoder_4')(decoder)
        decoder = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='decoder_5')(decoder)
        decoder = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='decoder_6')(decoder)
        decoder = layers.Reshape((NUM_FRAMES, NUM_BINS), name='decoder')(decoder)

        net_outputs = [conv1_aux, conv2_aux, conv3_aux, conv4_aux, conv5_aux, decoder]

        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=5e-1),
                           loss=Net.mse,
                           loss_weights={'conv1_aux': 0.1, 'conv2_aux': 0.2, 'conv3_aux': 0.3, 'conv4_aux': 0.4, 'conv5_aux': 0.6, 'decoder': 1})

    def construct_for_clustering(self):
        net_input = Net.input_layer()
        model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(net_input)

        model = ConvBN(64, (11, 9), strides=(5, 4), padding='same', name=1)(model, training=False)
        model = ConvBN(128, (9, 5), strides=(4, 2), padding='same', name=2)(model, training=False)
        model = ConvBN(256, (9, 5), strides=(4, 2), padding='same', name=3)(model, training=False)
        model = ConvBN(384, (5, 5), strides=(2, 2), padding='same', name=4)(model, training=False)
        model = ConvBN(512, (5, 5), strides=(2, 2), padding='same', name=5)(model, training=False)
        model = ConvBN(768, (5, 5), strides=(2, 2), padding='same', name=6)(model, training=False)

        embedded = layers.Reshape((NUM_FRAMES * NUM_BINS * 3 // 320,), name='embedded')(model)

        decoder = ConvTransposeBN(512, (3, 3), strides=(2, 2), padding='same', name='decoder_1')(model, training=False)
        decoder = ConvTransposeBN(384, (3, 3), strides=(2, 2), padding='same', name='decoder_2')(decoder, training=False)
        decoder = ConvTransposeBN(256, (3, 3), strides=(2, 2), padding='same', name='decoder_3')(decoder, training=False)
        decoder = ConvTransposeBN(128, (5, 3), strides=(4, 2), padding='same', name='decoder_4')(decoder, training=False)
        decoder = ConvTransposeBN(64, (5, 3), strides=(4, 2), padding='same', name='decoder_5')(decoder, training=False)
        decoder = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='decoder_6')(decoder)
        decoder = layers.Reshape((NUM_FRAMES, NUM_BINS), name='decoder')(decoder)

        net_outputs = [embedded, decoder]

        spread = Net.spreader(Net.BATCH_SIZE_CLUSTER)

        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-2),
                           loss={'embedded': spread, 'decoder': Net.mse},
                           loss_weights={'embedded': 0.05, 'decoder': 1})

    def construct_for_embedding(self):
        net_input = Net.input_layer()
        model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(net_input)

        model = ConvBN(64, (11, 9), strides=(5, 4), padding='same', name='1')(model, training=False)
        model = ConvBN(128, (9, 5), strides=(4, 2), padding='same', name='2')(model, training=False)
        model = ConvBN(256, (9, 5), strides=(4, 2), padding='same', name='3')(model, training=False)
        model = ConvBN(384, (5, 5), strides=(2, 2), padding='same', name='4')(model, training=False)
        model = ConvBN(512, (5, 5), strides=(2, 2), padding='same', name='5')(model, training=False)
        model = ConvBN(768, (5, 5), strides=(2, 2), padding='same', name='6')(model, training=False)

        net_output = layers.Reshape((NUM_FRAMES * NUM_BINS * 3 // 320,), name='embedded')(model)

        self.model = models.Model(inputs=net_input, outputs=net_output)

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=True)

    def predict(self, x):
        return self.model.predict(x)

    def pretrain(self):
        def __slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, {'conv1_aux': new_data, 'conv2_aux': new_data, 'conv3_aux': new_data, 'conv4_aux': new_data, 'conv5_aux': new_data, 'decoder': new_data}

        train_dataset = tf.data.Dataset.from_generator(
            train_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS])
        )
        #train_dataset = train_dataset.shuffle(Net.BATCH_SIZE_PRETRAIN * 4)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(Net.BATCH_SIZE_PRETRAIN)
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE_PRETRAIN * 4)

        #train_dataset = Net.get_pretrain_dataset()

        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-pretrain.hdf5'),
            save_weights_only=True
        )

        tensorboard_callback = callbacks.TensorBoard(
            log_dir='logs',
            write_graph=False,
            profile_batch=5
        )

        self.model.fit(x=train_dataset,
                       epochs=Net.NUM_EPOCHS,
                       steps_per_epoch=Net.STEPS_PER_EPOCH,
                       callbacks=[checkpoint_callback,
                                  tensorboard_callback],
                       verbose=1,
                       use_multiprocessing=True)

    def cluster(self):
        def __slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, {'embedded': np.zeros(768), 'decoder': new_data}

        train_dataset = tf.data.Dataset.from_generator(
            train_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS]),
            args=(Net.BATCH_SIZE_CLUSTER,)
        )
        train_dataset = train_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(Net.BATCH_SIZE_CLUSTER)
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE_CLUSTER * 4)

        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-cluster.hdf5'),
            save_weights_only=True,
            period=5
        )

        tensorboard_callback = callbacks.TensorBoard(
            log_dir='logs',
            write_graph=False,
            profile_batch=5
        )

        self.model.fit(x=train_dataset,
                       epochs=Net.NUM_EPOCHS,
                       callbacks=[checkpoint_callback,
                                  tensorboard_callback],
                       verbose=1,
                       use_multiprocessing=True)


def main():
    net = Net()

    net.construct_for_pretraining()
    tf.keras.utils.plot_model(net.model, to_file='model_pretrain.png', show_shapes=True)
    net.construct_for_pretraining()
    net.pretrain()

    net.construct_for_clustering()
    tf.keras.utils.plot_model(net.model, to_file='model_cluster.png', show_shapes=True)

    net.construct_for_embedding()
    tf.keras.utils.plot_model(net.model, to_file='model_embed.png', show_shapes=True)


if __name__ == '__main__':
    main()
