import pickle
import os
from collections import namedtuple
from typing import List, Dict

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import yaml

import constants
import spectrogram

Section = namedtuple('Section', ['name', 'start', 'end', 'rating'])
SectionData = namedtuple('SectionData', ['rating', 'data'])

all_section_data = []


def generate_data(y, fs, sections: List[Section]) -> Dict[str, np.ndarray]:
    all_data = {}
    for section in sections:
        start, end = section.start, section.end
        if (end - start) * fs >= constants.NUM_SAMPLES:
            data = y[start * fs: end * fs]
        else:
            more_required = constants.NUM_SAMPLES - (end - start) * fs
            start_required = more_required // 2
            end_required = more_required - start_required
            if start_required > start * fs:
                data = y[:constants.NUM_SAMPLES]
            elif end_required > len(y) - end * fs:
                data = y[-constants.NUM_SAMPLES:]
            else:
                data = y[start * fs - start_required: end * fs + end_required]

        if len(y.shape) == 1:
            all_data[section.name] = spectrogram.compute_spectrogram(data, fs)
        else:
            all_data[f'{section.name}_1'] = spectrogram.compute_spectrogram(np.asfortranarray(data[:, 0]), fs)
            all_data[f'{section.name}_2'] = spectrogram.compute_spectrogram(np.asfortranarray(data[:, 1]), fs)
    return all_data


def normalize(rating, std=0.16):
    return rating + np.random.normal(loc=0, scale=std)


def load_data_to_write(y, fs, sections):
    data = generate_data(y, fs, sections)
    if len(y.shape) == 1:
        for section in sections:
            section_data = data[section.name]
            all_section_data.append(SectionData(normalize(section.rating), section_data))
    else:
        for section in sections:
            section_data1 = data[f'{section.name}_1']
            section_data2 = data[f'{section.name}_2']
            all_section_data.append(SectionData(normalize(section.rating), section_data1))
            all_section_data.append(SectionData(normalize(section.rating), section_data2))


def do_write(file):
    global all_section_data

    num_data_points = 0
    with open(file, 'wb') as f:
        for section in all_section_data:
            pickle.dump(section, f)
            num_data_points += (len(section.data) - constants.NUM_FRAMES + 1)

    all_section_data = []
    return num_data_points


def load(file):
    fs, y = wavfile.read(file)
    y = np.divide(y, 32768)
    return y, fs


def generate():
    """
    Generate song data grouped by album
    """
    with open(os.path.join(os.path.dirname(__file__), 'data', 'mapping.yaml')) as mapping:
        all_data = yaml.safe_load(mapping)

    total_num_data_points = 0
    for artist in all_data:
        albums = all_data[artist]
        for album in albums:
            output_file = os.path.join(os.path.dirname(__file__), 'data', f'{artist}.{album}')
            songs = albums[album]
            for song in songs:
                song_file = os.path.join(os.path.dirname(__file__), '..', 'data', artist, album, f'{song}.wav')
                prefix = f'{artist}.{album}.{song}'
                sections = songs[song]
                y, fs = load(song_file)
                load_data_to_write(y, fs, [
                    Section(f"{prefix}.{section['name']}", section['start'], section['end'], section['rating'])
                    for section in sections
                ])
            total_num_data_points += do_write(output_file)
            
    print(f'Total number of data points (including augmentations): {total_num_data_points}')


def data_loader():
    with open(os.path.join(os.path.dirname(__file__), 'data', 'mapping.yaml')) as mapping:
        data = yaml.safe_load(mapping)
        filenames = [f'{artist}.{album}' for artist in data for album in data[artist]]

    np.random.shuffle(filenames)

    for filename in filenames:
        with open(os.path.join(os.path.dirname(__file__), 'data', filename), 'rb') as f:
            try:
                while True:
                    section_data = pickle.load(f)
                    yield section_data.data, section_data.rating
            except EOFError:
                pass


def random_slice(data):
    data_offset = tf.random.uniform([], 0, len(data) - constants.NUM_FRAMES + 1, dtype=tf.int32)
    return data[data_offset: data_offset + constants.NUM_FRAMES]


if __name__ == '__main__':
    generate()
