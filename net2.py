import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks
import numpy as np
from deprecation import deprecated

from constants import *

tf.compat.v1.disable_eager_execution()

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        #tf.config.experimental.set_virtual_device_configuration(
        #    gpu,
        #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=950)])
except RuntimeError as e:
    print(e)'''
    
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')

    
'''def batched_random_slice(data_batch, len_batch):
    b, m, n = data_batch.shape
    start_indices = tf.cast(tf.random.uniform((b,), 0, len_batch, dtype=tf.float64), tf.int64)
    adjusted_start_indices = start_indices + m * np.arange(b)
    data_batch = tf.reshape(data_batch, [-1])
    s = data_batch.numpy().strides[0]
    out = np.lib.stride_tricks.as_strided(data_batch, (b * m - 1, NUM_FRAMES, n), (s * n, s * n, s))[adjusted_slice_indices]
    #data_batch.shape = b, m, n
    return tf.convert_to_tensor(out)
    
    
def tf_batched_random_slice(data_batch, len_batch):
    batch_size = len_batch.shape[0]
    data = tf.py_function(batched_random_slice, [data_batch, len_batch], tf.float64)
    data.set_shape((batch_size, NUM_FRAMES, NUM_BINS))
    return data, data'''


def pretrain_train_data_loader():
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'train'), 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass
                
                
def pretrain_val_data_loader():
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'validation'), 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass
            
            
def cluster_train_data_loader(batch_size):
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'train'), 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                for _ in range(batch_size):
                    yield data
        except EOFError:
            pass
            
            
def cluster_val_data_loader(batch_size):
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'validation'), 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                for _ in range(batch_size):
                    yield data
        except EOFError:
            pass


class Net:
    RESIDUAL_TOWER_SIZE = 4
    
    NUM_EPOCHS = 10000
    STEPS_PER_EPOCH = 256
    BATCH_SIZE = 8
        
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
            return tf.reduce_sum(tf.math.square(y_pred - y_avg))
        return spread
        
    @staticmethod
    def mse(y_true, y_pred):
        """
        MSE loss function for 2D model output (since the built-in
           keras MSE doesn't behave the way we want it to)
        """
        return tf.reduce_sum(tf.math.square(y_pred - y_true), axis=[1, 2])
        
    @staticmethod
    def input_layer():
        return layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')
        
    @staticmethod
    def residual_block(input, training=True, encode=None, index=None):
        if encode is not None and index is not None:
            encode = 'enc' if encode else 'dec'
        
            model = layers.Conv1D(256, 3, padding='same',
                                  name=f'{encode}_res_conv1d_{2 * index + 1}')(input)
            model = layers.BatchNormalization(axis=2,
                                              name=f'{encode}_res_bn_{2 * index + 1}')(model, training=training)
            model = layers.ReLU()(model)
            model = layers.Conv1D(256, 3, padding='same',
                                  name=f'{encode}_res_conv1d_{2 * index + 2}')(model)
            model = layers.BatchNormalization(axis=2,
                                              name=f'{encode}_res_bn_{2 * index + 2}')(model, training=training)
            model = layers.Add()([model, input])
            model = layers.ReLU()(model)
            return model
        
    @staticmethod
    def construct_encoder(model, training=True):
        model = layers.Conv1D(256, 3, padding='same', name='enc_conv1d_1')(model)
        model = layers.BatchNormalization(axis=2, name='enc_bn_1')(model, training=training)
        model = layers.ReLU()(model)
        
        for i in range(Net.RESIDUAL_TOWER_SIZE):
            model = Net.residual_block(model, training=training, encode=True, index=i)
            
        model = layers.Conv1D(1, 1, padding='same', name='enc_conv1d_2')(model)
        model = layers.BatchNormalization(axis=2, name='enc_bn_2')(model, training=training)
        model = layers.ReLU()(model)
        model = layers.Reshape((NUM_FRAMES,))(model)
        model = layers.Dense(256, activation='relu', name='enc_dense_1')(model)
        model = layers.Dense(128, activation='relu', name='embedded')(model)
        
        return model
        
    @staticmethod
    def construct_decoder(model, training=True):
        model = layers.Dense(256, activation='relu', name='dec_dense_1')(model)
        model = layers.Dense(NUM_FRAMES, activation='relu', name='dec_dense_2')(model)
        model = layers.Reshape((NUM_FRAMES, 1))(model)
        model = layers.Conv1D(256, 1, padding='same', name='dec_conv1d_1')(model)
        model = layers.BatchNormalization(axis=2, name='dec_bn_1')(model, training=training)
        model = layers.ReLU()(model)
        
        for i in range(Net.RESIDUAL_TOWER_SIZE):
            model = Net.residual_block(model, training=training, encode=False, index=i)
            
        model = layers.Conv1D(NUM_BINS, 3, padding='same', name='dec_conv1d_2')(model)
        model = layers.BatchNormalization(axis=2, name='dec_bn_2')(model, training=training)
        model = layers.ReLU(name='reconstruction')(model)
        
        return model
        
    def construct_for_pretraining(self, training=True):
        net_input = Net.input_layer_2d()
        code = Net.construct_encoder_2d(net_input)
        net_output = Net.construct_decoder_2d(code)

        self.model = models.Model(net_input, net_output)
        self.model.summary()
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-1),
                           loss=Net.mse,
                           metrics=[Net.mse])
                           
    def construct_for_clustering(self):
        spread = Net.spreader(Net.BATCH_SIZE)
    
        net_input = Net.input_layer()
        code = Net.construct_encoder(net_input)
        net_output = Net.construct_decoder(code)
        
        self.model = models.Model(inputs=net_input,
                                  outputs=[code, net_output])
        self.model.summary()
        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-1),
                           loss={'embedded': spread, 'reconstruction': 'mse'},
                           metrics={'embedded': spread, 'reconstruction': 'mse'},
                           loss_weights={'embedded': 0.1, 'reconstruction': 1})
        
    def construct_for_embedding(self):
        net_input = Net.input_layer()
        code = Net.construct_encoder(net_input)
        
        self.model = models.Model(net_input, code)
        self.model.summary()
                           
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        
    def predict(self, x):
        return self.model.predict(x)
        
    @staticmethod
    def input_layer_2d():
        return layers.Input(shape=(NUM_FRAMES, NUM_BINS))
        
    @staticmethod
    def residual_block_2d(input):
        model = layers.Conv2D(16, (3, 3), padding='same')(input)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        model = layers.Conv2D(16, (3, 3), padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.Add()([model, input])
        model = layers.ReLU()(model)
        return model
        
    @staticmethod
    def construct_encoder_2d(model):
        model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(model)
        model = layers.Conv2D(16, (3, 3), padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        
        for _ in range(Net.RESIDUAL_TOWER_SIZE):
            model = Net.residual_block_2d(model)
            
        model = layers.Conv2D(1, (1, 1), padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        model = layers.Reshape((NUM_FRAMES, NUM_BINS))(model)
        model = layers.Conv1D(1, 1, padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        model = layers.Reshape((NUM_FRAMES,))(model)
        model = layers.Dense(256, activation='relu')(model)
        model = layers.Dense(128, activation='relu')(model)
        
        return model
        
    @staticmethod
    def construct_decoder_2d(model):
        model = layers.Dense(256, activation='relu')(model)
        model = layers.Dense(NUM_FRAMES, activation='relu')(model)
        model = layers.Reshape((NUM_FRAMES, 1))(model)
        model = layers.Conv1D(NUM_BINS, 1, padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(model)
        model = layers.Conv2D(16, (1, 1), padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        
        for _ in range(Net.RESIDUAL_TOWER_SIZE):
            model = Net.residual_block_2d(model)
            
        model = layers.Conv2D(1, (3, 3), padding='same')(model)
        model = layers.BatchNormalization(axis=-1)(model)
        model = layers.ReLU()(model)
        model = layers.Reshape((NUM_FRAMES, NUM_BINS))(model)
        
        return model
                           
    def construct_2d(self):
        net_input = Net.input_layer_2d()
        code = Net.construct_encoder_2d(net_input)
        net_output = Net.construct_decoder_2d(code)

        self.model = models.Model(net_input, net_output)
        self.model.summary()
        self.model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4),
                           loss='mse',
                           metrics=['mse'])
       
    @deprecated
    def train(self):
        def __slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, new_data
            
        '''train_dataset = tf.data.Dataset.from_generator(
            train_data_loader2,
            output_types=(tf.float64, tf.float64),
            output_shapes=(tf.TensorShape([None, NUM_BINS]), tf.TensorShape([]))
        )
        #train_dataset = train_dataset.cache()  # DANGEROUS!!!
        train_dataset = train_dataset.repeat()
        #train_dataset = train_dataset.shuffle(Net.BATCH_SIZE * 8, reshuffle_each_iteration=True)
        train_dataset = train_dataset.padded_batch(Net.BATCH_SIZE, ((None, NUM_BINS), ()))
        train_dataset = train_dataset.map(
            tf_batched_random_slice,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE * 4)'''
    
        train_dataset = tf.data.Dataset.from_generator(
            train_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS])
        )
        #train_dataset = train_dataset.cache()  # DANGEROUS!!!
        #train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(Net.BATCH_SIZE * 4, reshuffle_each_iteration=True)
        train_dataset = train_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(Net.BATCH_SIZE)
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE * 4)
        
        val_dataset = tf.data.Dataset.from_generator(
            val_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS])
        )
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(Net.BATCH_SIZE)
        val_dataset = val_dataset.prefetch(Net.BATCH_SIZE * 4)
        
        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-2b-1d-epoch{epoch:05d}-lr=1e-4.hdf5'),
            period=10
        )
        
        weights_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-2b-1d-epoch{epoch:05d}-lr=1e-4.hdf5'),
            save_weights_only=True,
            period=10
        )
        
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='loss',   # monitor train loss instead of validation loss
            factor=0.5,
            patience=80,
            verbose=1,
            cooldown=20
        )
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir='logs',
            write_graph=False,
            profile_batch=5
        )
        
        self.model.fit(x=train_dataset,
                       epochs=Net.NUM_EPOCHS,
                       #steps_per_epoch=Net.STEPS_PER_EPOCH,
                       callbacks=[checkpoint_callback,
                                  weights_callback,
                                  reduce_lr_callback,
                                  tensorboard_callback],
                       verbose=1,
                       validation_data=val_dataset,
                       validation_steps=4,
                       use_multiprocessing=True)
                       
    def pretrain(self):
        def __slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, new_data
    
        train_dataset = tf.data.Dataset.from_generator(
            pretrain_train_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS])
        )
        train_dataset = train_dataset.shuffle(Net.BATCH_SIZE * 4, reshuffle_each_iteration=True)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(Net.BATCH_SIZE)
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE * 4)
        
        '''val_dataset = tf.data.Dataset.from_generator(
            pretrain_val_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS])
        )
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(Net.BATCH_SIZE)
        val_dataset = val_dataset.prefetch(Net.BATCH_SIZE * 4)'''
        
        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-pretrain-epoch{epoch:05d}.hdf5'),
            period=2
        )
        
        weights_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-pretrain-epoch{epoch:05d}.hdf5'),
            save_weights_only=True,
            period=2
        )
        
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='loss',   # monitor train loss instead of validation loss
            factor=0.5,
            patience=5,   #
            verbose=1,
            cooldown=1    #
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
                                  weights_callback,
                                  reduce_lr_callback,
                                  tensorboard_callback],
                       verbose=1,
                       #validation_data=val_dataset,
                       #validation_steps=8,
                       #validation_freq=10,
                       use_multiprocessing=True)
                       
    def cluster(self):
        def __slice_n_dice(data):
            data_offset = tf.random.uniform([], 0, len(data) - NUM_FRAMES + 1, dtype=tf.int32)
            new_data = data[data_offset: data_offset + NUM_FRAMES]
            return new_data, {'embedded': np.zeros(128) , 'reconstruction': new_data}
        
        train_dataset = tf.data.Dataset.from_generator(
            cluster_train_data_loader,
            output_types=tf.float64,
            output_shapes=tf.TensorShape([None, NUM_BINS]),
            args=(Net.BATCH_SIZE,)
        )
        train_dataset = train_dataset.map(__slice_n_dice, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(Net.BATCH_SIZE)
        train_dataset = train_dataset.prefetch(Net.BATCH_SIZE * 4)
        
        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint-cluster-2b-1d-epoch{epoch:05d}-adam-lr=1e-1.hdf5'),
            period=10
        )
        
        weights_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-cluster-2b-1d-epoch{epoch:05d}-adam-lr=1e-1.hdf5'),
            save_weights_only=True,
            period=10
        )
        
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='loss',   # monitor train loss instead of validation loss
            factor=0.5,
            patience=80,
            verbose=1,
            min_delta=1e-5,
            cooldown=20
        )
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir='logs',
            write_graph=False,
            profile_batch=5
        )
        
        self.model.fit(x=train_dataset,
                       epochs=Net.NUM_EPOCHS,
                       callbacks=[checkpoint_callback,
                                  weights_callback,
                                  reduce_lr_callback,
                                  tensorboard_callback],
                       verbose=1,
                       use_multiprocessing=True)


def main():
    net = Net()
    net.construct_for_pretraining()
    net.pretrain()
    
    #net = Net()
    #net.construct_for_clustering()
    #net.load_weights('checkpoints/weights-2b-1d-epoch00830-adam-lr=1e-1.hdf5')
    #net.cluster()


if __name__ == '__main__':
    main()
