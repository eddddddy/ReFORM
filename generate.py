"""
Utility module to generate normalized mel spectrograms
   from input wav files
"""

import pickle
import os
import multiprocessing
import queue
import pathlib

import numpy as np
from scipy.io import wavfile
import yaml

import spectrogram

VALIDATION_RATIO = 0.00000001
DATA_PATH = "D:/Data"

TRAIN_OUTPUT_FILEPATH = os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'train')
VALIDATION_OUTPUT_FILEPATH = os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'validation')
ALL_OUTPUT_FILEPATH = os.path.join(os.path.dirname(__file__), 'spectrogram_data', 'all')

NUM_THREADS = 8


def load(file):
    fs, y = wavfile.read(file)
    y = np.divide(y, 32768)
    return y, fs


def get_nth_train_filepath(n):
    return os.path.join(os.path.dirname(__file__), 'spectrogram_data', f'train{n}')


def generate_worker(song_queue, data_queue):
    try:
        while True:
            song_info = song_queue.get(True)
            if not song_info:
                return

            artist, album, song = song_info

            song_file = os.path.join(DATA_PATH, artist, album, f'{song}.wav')
            y, fs = load(song_file)
            if len(y.shape) == 1:
                raise ValueError(f"Mono is not supported: {song}.wav")

            spectrogram1 = spectrogram.compute_normal_spectrogram(y[:, 0], fs)
            spectrogram2 = spectrogram.compute_normal_spectrogram(y[:, 1], fs)

            data_queue.put((artist, album, song, spectrogram1, spectrogram2))
    except queue.Empty:
        return


def generate():
    with open(os.path.join(os.path.dirname(__file__), 'mapping.yaml')) as mapping:
        all_data = yaml.safe_load(mapping)

    high = int(1 / VALIDATION_RATIO)

    train_data = []
    validation_data = []
    data = []

    workers = []
    song_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()

    pathlib.Path(os.path.join(os.path.dirname(__file__), 'spectrogram_data')).mkdir(exist_ok=True)

    try:
        train_file = open(TRAIN_OUTPUT_FILEPATH, 'wb')
        validation_file = open(VALIDATION_OUTPUT_FILEPATH, 'wb')
        all_file = open(ALL_OUTPUT_FILEPATH, 'wb')

        for artist in all_data:
            albums = all_data[artist]
            for album in albums:
                songs = albums[album]
                for song in songs:
                    song_queue.put((artist, album, song['name']))

        for _ in range(NUM_THREADS):
            workers.append(multiprocessing.Process(target=generate_worker, args=(song_queue, data_queue)))

            # why is this here? won't one of the threads just exit right away?
            song_queue.put(None)

        for worker in workers:
            worker.start()

        while True:
            try:
                artist, album, song, spectrogram1, spectrogram2 = data_queue.get(False)
                song_str = f'{artist}.{album}.{song}'
                (validation_data if np.random.randint(1, high) == 1 else train_data).append(spectrogram1)
                (validation_data if np.random.randint(1, high) == 1 else train_data).append(spectrogram2)
                data.extend([spectrogram1, song_str, spectrogram2, song_str])

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
            except queue.Empty:
                for worker in workers:
                    if worker.is_alive():
                        break
                else:
                    break

        for song_data in train_data:
            pickle.dump(song_data, train_file)

        for song_data in validation_data:
            pickle.dump(song_data, validation_file)

        for d in data:
            pickle.dump(d, all_file)

    finally:
        train_file.close()
        validation_file.close()
        all_file.close()


def split_data():
    """
    Create NUM_THREADS equal-sized training files that collectively contain
    all the training data in the training output file of a generate() call
    """
    try:
        train_files = [open(get_nth_train_filepath(i), 'wb') for i in range(NUM_THREADS)]

        with open(TRAIN_OUTPUT_FILEPATH, 'rb') as f:
            i = 0
            try:
                while True:
                    pickle.dump(pickle.load(f), train_files[i])
                    i = (i + 1) % NUM_THREADS
            except EOFError:
                pass
    finally:
        for train_file in train_files:
            train_file.close()


if __name__ == '__main__':
    generate()
    #split_data()
