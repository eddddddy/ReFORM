import os
import pickle

import yaml
import tensorflow as tf
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt

import constants
import net2


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


def get_encoder(checkpoint_path):
    net = net2.Net()
    net.construct_for_embedding()
    net.load_weights(checkpoint_path)
    return net
    
    
def data_loader():
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'all'), 'rb') as f:
        try:
            while True:
                spectrogram_data = pickle.load(f)
                name = pickle.load(f)
                yield spectrogram_data, name
        except EOFError:
            pass
    
    
def get_latent_vectors(encoder):
    vectors = None
    labels = []
    
    def map_fn(data, name):
        labels.append(name)
        offset = np.random.randint(0, len(data) - constants.NUM_FRAMES + 1)
        return data[offset: offset + constants.NUM_FRAMES]
        
    def multiplicity_fn(data, name):
        return 16 if name == 'palette' else 4

    batcher = Batcher(data_loader(), 8, multiplicity_fn, map_fn)
    
    while True:
        batch = batcher.next()
        
        if batch is None:
            break
        
        predictions = encoder.predict(batch)
        
        if vectors is None:
            vectors = predictions
        else:
            vectors = np.vstack([vectors, predictions])
          
    return vectors, labels
    
    
def main():
    print("Encoding vectors...")
    encoder = get_encoder('checkpoints/weights-cluster-2b-1d-epoch01190-adam-lr=1e-1.hdf5')
    vectors, labels = get_latent_vectors(encoder)
    labels = np.array(labels, dtype=object)
    
    print("Running t-SNE...")
    tsne = sklearn.manifold.TSNE(
        n_components=2,
        #early_exaggeration=24.0,
        learning_rate=100.0,
        n_iter=2000, 
        n_iter_without_progress=500, 
        min_grad_norm=0
    )
    projected_vectors = tsne.fit_transform(vectors)
    
    print("Plotting...")
    plt.figure()
    plt.scatter(projected_vectors[labels != 'palette', 0], projected_vectors[labels != 'palette', 1], c='b')
    plt.scatter(projected_vectors[labels == 'palette', 0], projected_vectors[labels == 'palette', 1], c='r')
    plt.show()
    
    
if __name__ == '__main__':
    main()
