"""
Utility module to generate normalized mel spectrograms
   from input wav files
"""

import pickle
import os

import numpy as np
from scipy.io import wavfile
import yaml

import constants
import spectrogram


VALIDATION_RATIO = 0.1


def load(file):
    fs, y = wavfile.read(file)
    y = np.divide(y, 32768)
    return y, fs
    
    
def get_counts(arr):
    return np.array([((arr > a) & (arr < (a + 0.1))).sum() for a in np.arange(0, 1, 0.1)])


def generate():
    with open(os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'mapping.yaml')) as mapping:
        all_data = yaml.safe_load(mapping)
    
    high = int(1 / VALIDATION_RATIO)
    
    train_data = []
    validation_data = []
    data = []
    train_output_file = os.path.join(os.path.dirname(__file__), 'experimental_data', 'train')
    validation_output_file = os.path.join(os.path.dirname(__file__), 'experimental_data', 'validation')
    all_output_file = os.path.join(os.path.dirname(__file__), 'experimental_data', 'all')
    
    cum_counts = np.zeros(10, dtype=np.int32)
    
    try:
        train_file = open(train_output_file, 'wb')
        validation_file = open(validation_output_file, 'wb')
        all_file = open(all_output_file, 'wb')
        
        for artist in all_data:
            albums = all_data[artist]
            for album in albums:
                songs = albums[album]
                for song in songs:
                    song_file = os.path.join(os.path.dirname(__file__), '..', 'data', artist, album, f'{song}.wav')
                    y, fs = load(song_file)
                    if len(y.shape) == 1:
                        raise ValueError(f"Mono is not supported: {song}.wav")
                    
                    spectrogram1 = spectrogram.compute_normal_spectrogram(y[:, 0], fs)
                    spectrogram2 = spectrogram.compute_normal_spectrogram(y[:, 1], fs)
                    (validation_data if np.random.randint(1, high) == 1 else train_data).append(spectrogram1)
                    (validation_data if np.random.randint(1, high) == 1 else train_data).append(spectrogram2)
                    data.extend([spectrogram1, song, spectrogram2, song])
                    
                    cum_counts += get_counts(spectrogram1) + get_counts(spectrogram2)
                    print(cum_counts)
                
                    if len(train_data) > 10:
                        for song_data in train_data:
                            pickle.dump(song_data, train_file)
                        train_data = []
                        
                    if len(validation_data) > 10:
                        for song_data in validation_data:
                            pickle.dump(song_data, validation_file)
                        validation_data = []
                            
                    if len(data) > 10:
                        for d in data:
                            pickle.dump(d, all_file)
                        data = []
    finally:
        train_file.close()
        validation_file.close()
        all_file.close()


if __name__ == '__main__':
    generate()
