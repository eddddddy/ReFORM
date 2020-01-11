import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, constraints

import embed

good_songs = {'heart', 'peach', 'untitled_song', 'station', 'fifth_finger'}


class Recommender():

    def __init__(self, baselines):
        self.model = None
        self.baselines = baselines

    def input_layer(self):
        return layers.Input(shape=(len(self.baselines),), name='input')

    def construct(self):
        net_input = self.input_layer()
        net_output = layers.Dense(1, activation='sigmoid', kernel_constraint=constraints.NonNeg)(net_input)

        self.model = models.Model(inputs=net_input, outputs=net_output)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                           loss='mse')


def main():
    heart = embed.load_embeddings('heart')
    similarities = embed.compute_similarities(heart)
    for similarity in similarities:
        if similarity[0] in good_songs:
            print(f'{similarity[0]} {similarity[1]} {similarities.index(similarity)}')


if __name__ == '__main__':
    main()
