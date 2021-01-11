import os
import pickle
import gc

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from constants import *
import generate
import net


def data_loader():
    with open(generate.ALL_OUTPUT_FILEPATH, 'rb') as f:
        try:
            while True:
                spectrogram_data = pickle.load(f)
                name = pickle.load(f)
                yield spectrogram_data, name
        except EOFError:
            pass


class Embedding:
    
    def __init__(self, model):
        self.embed_data = {}
        
        if isinstance(model, net.Net):
            self.model = model
        elif isinstance(model, str):
            m = net.Net()
            m.construct_encoder()
            m.load_weights(model)
            self.model = m
        else:
            raise ValueError("model must be a Net or a str")
    
    @staticmethod
    def __pairwise_ed(a, b):
        ba, bb = a.shape[0], b.shape[0]
        sqr_norm_a = np.sum(np.power(a, 2), axis=1).reshape(1, ba)
        sqr_norm_b = np.sum(np.power(b, 2), axis=1).reshape(bb, 1)
        inner_prod = b @ a.transpose()
        tile1 = np.tile(sqr_norm_a, (bb, 1))
        tile2 = np.tile(sqr_norm_b, (1, ba))
        return tile1 + tile2 - 2 * inner_prod
    
    @staticmethod
    def __smoothen(points, boundary='periodic', num=100):
        x, y = points[:, 0], points[:, 1]

        t = np.zeros(x.shape)
        t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]
        nt = np.linspace(0, 1, num)

        x = CubicSpline(t, x, bc_type=boundary)(nt)
        y = CubicSpline(t, y, bc_type=boundary)(nt)

        return np.concatenate([x, y]).reshape(2, -1).transpose()
    
    def find_data(self, name):
        # TODO: make this work for stereo data (right now it ignores the first channel)
        data_gen = data_loader()
        if isinstance(name, str):
            name = [name]
        
        data = {}
        if isinstance(name, list):
            try:
                while True:
                    x, n = next(data_gen)
                    if n in name:
                        data[n] = x
            except StopIteration:
                pass
        else:
            try:
                while True:
                    x, n = next(data_gen)
                    data[n] = x
            except StopIteration:
                pass
                
        name, data = zip(*list(data.items()))
        return list(name), list(data)
    
    def calculate(self, mel_spec=None, name=None, num_points=512, trim=100, window=NUM_FRAMES, batch_size=32):
    
        def get_embedding(data):
            data = data.copy()
            if trim:
                data = data[trim:-trim]
            s0, s1 = data.strides
            s = (data.shape[0] - NUM_FRAMES) // (num_points - 1)
            strided_data = as_strided(data, shape=(num_points, window, NUM_BINS), strides=(s * s0, s0, s1))
            return self.model.predict(strided_data, batch_size=batch_size)
            
        if mel_spec:
            if name:
                self.embed_data[name] = get_embedding(mel_spec)
            else:
                raise ValueError("Name must be provided if mel_spec is provided")
        else:
            names, data = self.find_data(name)
            for i, name in enumerate(names):
                self.embed_data[name] = get_embedding(data[i])
                gc.collect()
                
    def similarity(self, name1, name2):
        if name1 not in self.embed_data:
            self.calculate(name1)
        if name2 not in self.embed_data:
            self.calculate(name2)
        
        a, b = self.embed_data[name1], self.embed_data[name2]
        sum_pairwise_dists = np.sum(Embedding.__pairwise_ed(a, b))
        a_sim = np.sum(Embedding.__pairwise_ed(a, a)) / sum_pairwise_dists
        b_sim = np.sum(Embedding.__pairwise_ed(b, b)) / sum_pairwise_dists
        return (a_sim + b_sim) / 2

    def plot(self, boundary='smooth', figsize=(16, 9)):
        if boundary not in [None, 'convex', 'smooth']:
            raise ValueError("Boundary must be None, convex, or smooth")
    
        embed_data = list(self.embed_data.items())
        projected = PCA(n_components=2, svd_solver='full').fit_transform(np.vstack([data for _, data in embed_data]))
        
        indices = [0] + [data.shape[0] for _, data in embed_data]
        for i in range(1, len(indices)):
            indices[i] = indices[i - 1] + indices[i]
        
        if boundary is not None:
            hulls = [ConvexHull(projected[indices[i]:indices[i + 1]]) for i in range(len(embed_data))]
            hulls = [np.concatenate([projected[indices[i]:indices[i + 1]][hulls[i].vertices],
                                     projected[indices[i]:indices[i + 1]][hulls[i].vertices[0]].reshape(1, -1)])
                     for i in range(len(embed_data))]
            if boundary == 'smooth':
                hulls = [Embedding.__smoothen(hull) for hull in hulls]
            
        plt.figure(figsize=figsize)
        for i, (name, _) in enumerate(embed_data):
            plt.scatter(projected[indices[i]:indices[i + 1], 0], projected[indices[i]:indices[i + 1], 1], label=name)
            if boundary is not None:
                plt.plot(hulls[i][:, 0], hulls[i][:, 1])
                
        plt.legend(loc='best')
        plt.show()
