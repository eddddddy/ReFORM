import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, callbacks

from constants import *
from generate import data_loader, random_slice, SectionData


class LRScheduler(callbacks.Callback):
    SMOOTHING = 0.1
    HORIZON = 30
    PATIENCE = 8

    def __init__(self):
        self.learning_rate = None
        self.moving_average = None
        self.epsilon = None
        self.last_lr_drop_epoch = None
        self.epochs_remaining = None

    def on_train_begin(self, logs=None):
        self.learning_rate = float(backend.get_value(self.model.optimizer.lr))
        self.moving_average = -1
        self.epsilon = 0.05
        self.last_lr_drop_epoch = 0
        self.epochs_remaining = LRScheduler.PATIENCE

    def update_learning_rate(self, epoch):
        self.learning_rate /= 10
        self.last_lr_drop_epoch = epoch
        self.epsilon = self.moving_average / 400
        self.epochs_remaining = LRScheduler.PATIENCE
        backend.set_value(self.model.optimizer.lr, self.learning_rate)
        print(f'Updating LR to {self.learning_rate} at start of epoch {epoch + 1}')

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            self.moving_average = logs['loss']
            return

        epoch_delta = min(LRScheduler.HORIZON, epoch - self.last_lr_drop_epoch)
        new_moving_average = (logs['loss'] + (epoch_delta - 1) * self.moving_average) / epoch_delta
        if abs(self.moving_average - new_moving_average) < self.epsilon:
            self.epochs_remaining -= 1
            if self.epochs_remaining == 0:
                self.update_learning_rate(epoch)
        else:
            self.epochs_remaining = LRScheduler.PATIENCE

        self.moving_average = new_moving_average
        print(f'Average loss at epoch {epoch}: {self.moving_average}')


class Net:
    NUM_EPOCHS = 100000
    SHUFFLE_BUFFER_SIZE = 32
    BATCH_SIZE = 4
    PREFETCH_BUFFER_SIZE = BATCH_SIZE * 2

    def __init__(self):
        self.model = None

    @staticmethod
    def l2_pool(x):
        x = backend.square(x)
        x = layers.GlobalAvgPool1D()(x)
        x = backend.sqrt(x)
        return x

    def construct(self):
        net_input = layers.Input(shape=(NUM_FRAMES, NUM_BINS))

        model = layers.Conv1D(256, 4, padding='same', activation='relu')(net_input)
        model = layers.MaxPool1D(4, padding='same')(model)
        model = layers.Conv1D(256, 4, padding='same', activation='relu')(model)
        model = layers.MaxPool1D(2, padding='same')(model)
        model = layers.Conv1D(256, 4, padding='same', activation='relu')(model)
        model = layers.MaxPool1D(2, padding='same')(model)
        model = layers.Conv1D(512, 4, padding='same', activation='relu')(model)
        mean_pool = layers.GlobalAvgPool1D()(model)
        max_pool = layers.GlobalMaxPool1D()(model)
        l2_pool = layers.Lambda(Net.l2_pool)(model)
        model = layers.Concatenate()([mean_pool, max_pool])
        model = layers.Dense(2048, activation='relu')(model)
        model = layers.Dropout(0.2)(model)
        model = layers.Dense(2048, activation='relu')(model)
        model = layers.Dropout(0.2)(model)
        model = layers.Dense(2048, activation='relu')(model)
        model = layers.Dropout(0.2)(model)

        net_output = layers.Dense(1, activation='relu')(model)

        self.model = models.Model(net_input, net_output)
        self.model.summary()
        self.model.compile(optimizer=optimizers.Nadam(learning_rate=1e-6),
                           loss='mse',
                           metrics=['mse'])

    def train(self):
        dataset = tf.data.Dataset.from_generator(data_loader, (tf.float64, tf.float64),
                                                 output_shapes=(tf.TensorShape([None, NUM_BINS]),
                                                                tf.TensorShape([])))
        dataset = dataset.shuffle(Net.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (random_slice(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(Net.BATCH_SIZE)

        self.model.fit(x=dataset,
                       epochs=Net.NUM_EPOCHS,
                       callbacks=[LRScheduler()],
                       verbose=0)


def main():
    net = Net()
    net.construct()
    net.train()


main()
