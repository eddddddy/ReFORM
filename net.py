import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks
import numpy as np

from constants import *
import generate
import serialize

#tf.compat.v1.disable_eager_execution()

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
    STEPS_PER_EPOCH = 64
    BATCH_SIZE_TRAIN = 16
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
    def get_train_dataset():
        def slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, new_data
        
        train_files = [serialize.get_nth_serialized_train_filepath(i) for i in range(generate.NUM_THREADS)]

        dataset = tf.data.Dataset.from_tensor_slices(train_files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=generate.NUM_THREADS,
            deterministic=False
        ).map(serialize.decode)
        
        dataset = dataset.repeat()
        dataset = dataset.shuffle(Net.BATCH_SIZE_TRAIN * 2)
        dataset = dataset.map(slice_n_dice)
        dataset = dataset.batch(Net.BATCH_SIZE_TRAIN)
        dataset = dataset.prefetch(Net.BATCH_SIZE_TRAIN)
        
        return dataset

    @staticmethod
    def input_layer():
        return layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')
        
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
        
    def construct_for_training(self):
        net_input = Net.input_layer()
        model = net_input
        
        model = Net.encoder_conv(model, 256, 11, 5, 'conv_1')
        model = Net.encoder_conv(model, 384, 9, 4, 'conv_2')
        model = Net.encoder_conv(model, 512, 7, 3, 'conv_3')
        model = Net.encoder_conv(model, 512, 5, 2, 'conv_4')
        model = Net.encoder_conv(model, 512, 5, 2, 'conv_5')
        model = Net.encoder_conv(model, 512, 3, 1, 'conv_6', padding='valid')
        model = Net.encoder_conv(model, 768, 2, 1, 'conv_7', padding='valid')
        
        model = layers.Reshape((768,))(model)
        model = layers.Dense(512, name='dense1')(model)
        model = layers.Dense(768, activation='relu', name='dense2')(model)
        model = layers.Reshape((1, 768))(model)
        
        model = Net.decoder_conv(model, 512, 3, 2, 'dconv_1')
        model = Net.decoder_conv(model, 512, 3, 2, 'dconv_2')
        model = Net.decoder_conv(model, 512, 5, 2, 'dconv_3')
        model = Net.decoder_conv(model, 512, 5, 2, 'dconv_4')
        model = Net.decoder_conv(model, 384, 7, 3, 'dconv_5')
        model = Net.decoder_conv(model, 256, 9, 4, 'dconv_6')
        
        model = layers.UpSampling1D(5)(model)
        model = layers.Conv1D(128, 11, activation='sigmoid', padding='same', name='dconv_7')(model)
        
        net_outputs = model
        
        self.model = models.Model(inputs=net_input, outputs=net_outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=3e-4), loss=Net.mse)

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
                       callbacks=[checkpoint_callback, tensorboard_callback],
                       verbose=1)
        
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

    net.construct_for_training()
    tf.keras.utils.plot_model(net.model, to_file='model_train.png', show_shapes=True)
    net.train()

    #net.construct_for_clustering()
    #tf.keras.utils.plot_model(net.model, to_file='model_cluster.png', show_shapes=True)

    #net.construct_for_embedding()
    #tf.keras.utils.plot_model(net.model, to_file='model_embed.png', show_shapes=True)


if __name__ == '__main__':
    main()
