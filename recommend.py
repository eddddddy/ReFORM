import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, constraints, regularizers
import yaml

import embed

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


class Recommender:
    NUM_EPOCHS = 10000
    STEPS_PER_EPOCH = 512
    BATCH_SIZE = 128

    def __init__(self):
        self.model = None
        self.baselines = Recommender.get_baselines()

    @staticmethod
    def get_baselines():
        with open(os.path.join(os.path.dirname(__file__), 'mapping.yaml')) as mapping:
            data = yaml.safe_load(mapping)
            baselines = []
            for artist in data:
                for album in data[artist]:
                    for song in data[artist][album]:
                        if song['rating'] == 2:
                            baselines.append(f'{artist}.{album}.{song["name"]}')
            return baselines

    def get_baseline_centroids(self):
        centroids = {}
        for baseline in self.baselines:
            centroids[baseline] = np.mean(embed.load_embeddings(baseline), axis=0)
        return centroids

    def get_dataset(self):
        input_data = []
        labels = []
        baseline_centroids = self.get_baseline_centroids()
        with open(os.path.join(os.path.dirname(__file__), 'mapping.yaml')) as mapping:
            data = yaml.safe_load(mapping)
            for artist in data:
                for album in data[artist]:
                    for song in data[artist][album]:
                        song_name = f'{artist}.{album}.{song["name"]}'

                        if song_name in self.baselines:
                            continue

                        similarities = []
                        embeddings = embed.load_embeddings(song_name)
                        for _, centroid in baseline_centroids.items():
                            similarities.append(np.abs(np.mean(embeddings, axis=0) - centroid))

                        input_data.append(-1 * np.array(similarities))
                        labels.append(song['rating'])

        return tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(np.array(input_data)),
            tf.data.Dataset.from_tensor_slices(labels)
        ))

    def input_layer(self):
        return layers.Input(shape=(len(self.baselines), 768), name='input')

    def construct(self):
        net_input = self.input_layer()
        net_output = layers.Conv1D(1, 1, padding='same', kernel_constraint=constraints.NonNeg(), use_bias=False, name='scale')(net_input)
        net_output = layers.Reshape((len(self.baselines),))(net_output)
        net_output = layers.Dense(1, activation='sigmoid', kernel_constraint=constraints.NonNeg(), kernel_regularizer=regularizers.l2(l=5),
                                  name='output')(net_output)

        self.model = models.Model(inputs=net_input, outputs=net_output)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-1),
                           loss='binary_crossentropy')

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=True)

    def predict(self, x):
        return self.model.predict(x)

    def train(self):
        train_dataset = self.get_dataset().repeat()
        train_dataset = train_dataset.batch(Recommender.BATCH_SIZE)
        train_dataset = train_dataset.prefetch(Recommender.BATCH_SIZE * 4)

        checkpoint_callback = callbacks.ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'weights-recommend.hdf5'),
            save_weights_only=True,
            period=80
        )

        self.model.fit(
            x=train_dataset,
            epochs=Recommender.NUM_EPOCHS,
            steps_per_epoch=Recommender.STEPS_PER_EPOCH,
            callbacks=[checkpoint_callback],
            verbose=1,
            use_multiprocessing=True
        )


def main():
    recommender = Recommender()
    recommender.construct()
    recommender.model.summary()
    recommender.train()


if __name__ == '__main__':
    main()
