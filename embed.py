import os
import pickle

import numpy as np

import constants
import net


class Batcher:

    def __init__(self, generator, batch_size, multiplicity_fn=None, map_fn=None):
        self.__generator = generator
        self.__batch_size = batch_size
        self.__multiplicity_fn = multiplicity_fn
        self.__map_fn = map_fn
        self.__current = None
        self.__remaining = 0

    def next(self):
        batched = None

        for _ in range(self.__batch_size):
            if self.__remaining > 0:
                data = self.__current
                self.__remaining -= 1
            else:
                try:
                    data = next(self.__generator)
                except StopIteration:
                    return batched

                self.__current = data
                if self.__multiplicity_fn:
                    self.__remaining = self.__multiplicity_fn(*data) - 1

            if self.__map_fn:
                data = self.__map_fn(*data)

            if batched is None:
                batched = np.expand_dims(data, axis=0)
            else:
                batched = np.vstack([batched, np.expand_dims(data, axis=0)])

        return batched


def save_embeddings(vectors, name):
    output_file = os.path.join(os.path.dirname(__file__), 'embed_data', name)
    with open(output_file, 'wb') as f:
        for i in range(len(vectors)):
            pickle.dump(vectors[i], f)


def load_embeddings(name):
    file = os.path.join(os.path.dirname(__file__), 'embed_data', name)
    vectors = None
    with open(file, 'rb') as f:
        try:
            while True:
                vector = pickle.load(f)
                if vectors is None:
                    vectors = np.expand_dims(vector, axis=0)
                else:
                    vectors = np.vstack([vectors, np.expand_dims(vector, axis=0)])
        except EOFError:
            return vectors


def get_encoder(checkpoint_path):
    model = net.Net()
    model.construct_for_embedding()
    model.load_weights(checkpoint_path)
    return model


def data_loader():
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'all'), 'rb') as f:
        try:
            while True:
                spectrogram_data = pickle.load(f)
                name = pickle.load(f)
                yield spectrogram_data, name
        except EOFError:
            pass


def determine_embeddings(encoder):
    vectors = None
    labels = []

    def multiplicity_fn(data, name):
        return 80

    def create_map_fn():
        total_samples = {}
        current_sample = {}

        def map_fn(data, name):
            labels.append(name)
            data = data[100:-100]

            if name not in total_samples:
                total_samples[name] = multiplicity_fn(data, name)
                current_sample[name] = 0

            offset = current_sample[name] * (len(data) - constants.NUM_FRAMES) // (total_samples[name] - 1)
            current_sample[name] = (current_sample[name] + 1) % total_samples[name]

            return data[offset: offset + constants.NUM_FRAMES]

        return map_fn

    batcher = Batcher(data_loader(), 8, multiplicity_fn, create_map_fn())

    while True:
        batch = batcher.next()

        if batch is None:
            break

        predictions = encoder.predict(batch)

        if vectors is None:
            vectors = predictions
        else:
            vectors = np.vstack([vectors, predictions])

    while labels:
        name = labels[0]
        last_index = len(labels) - labels[::-1].index(name)
        save_embeddings(vectors[:last_index], name)

        labels = labels[last_index:]
        vectors = vectors[last_index:]


def get_latent_vectors():
    vectors = None
    labels = []
    for name in os.listdir(os.path.join(os.path.dirname(__file__), 'embed_data')):
        embeddings = load_embeddings(name)
        if vectors is None:
            vectors = embeddings
        else:
            vectors = np.vstack([vectors, embeddings])
        labels.extend([name] * len(embeddings))
    return vectors, labels


def sum_pairwise_distances(v1, v2):
    u1 = np.lib.stride_tricks.as_strided(
        v1,
        (v1.shape[0], v2.shape[0], v1.shape[1]),
        (v1.strides[0], 0, v1.strides[1])).reshape((-1, v1.shape[1]))
    u2 = np.lib.stride_tricks.as_strided(
        v2,
        (v1.shape[0], v2.shape[0], v2.shape[1]),
        (0, v2.strides[0], v2.strides[1])).reshape((-1, v2.shape[1]))

    return np.sum(np.power(np.sum(np.power(np.abs(u1 - u2), 0.2), axis=1), 5))


def centroid_distance(v1, v2):
    return np.sqrt(np.sum(np.power(np.abs(np.mean(v1, axis=0) - np.mean(v2, axis=0)), 2)))


def compute_similarity(vectors1, vectors2):
    return centroid_distance(vectors1, vectors2)
    # norm = (sum_pairwise_distances(vectors1, vectors1) + sum_pairwise_distances(vectors2, vectors2)) / 2
    # return norm / total


def compute_similarities(vectors):
    similarities = []
    for name in os.listdir(os.path.join(os.path.dirname(__file__), 'embed_data')):
        similarities.append([name, compute_similarity(vectors, load_embeddings(name))])
    similarities = sorted(similarities, key=lambda p: p[0])
    return similarities
