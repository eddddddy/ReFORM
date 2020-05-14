import os
import pickle

import tensorflow as tf
import numpy as np

import constants
import generate


SERIALIZED_TRAIN_FILE = os.path.join(os.path.dirname(__file__), 'spectrogram_data', f'train.tfrecord')


def get_nth_serialized_train_filepath(n):
    return os.path.join(os.path.dirname(__file__), 'spectrogram_data', f'train{n}.tfrecord')

   
def to_feature_list(array_2d):
    float_lists = [tf.train.Feature(float_list=tf.train.FloatList(value=row)) for row in array_2d]
    return tf.train.FeatureList(feature=float_lists)

    
def serialize(array):
    feature_list = {'spec': to_feature_list(array)}
    example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    
    return example.SerializeToString()


def write_tfrecords():
    for i in range(generate.NUM_THREADS):
        raw_filepath = generate.get_nth_train_filepath(i)
        serialized_filepath = get_nth_serialized_train_filepath(i)
        with open(raw_filepath, 'rb') as raw_file:
            with tf.io.TFRecordWriter(serialized_filepath) as writer:
                try:
                    while True:
                        data = pickle.load(raw_file)
                        writer.write(serialize(data))
                except EOFError:
                    pass


feature_description = {
    'spec': tf.io.FixedLenSequenceFeature(constants.NUM_BINS, tf.float32)
}

def decode(example):
    return tf.io.parse_single_sequence_example(example, sequence_features=feature_description)[1].get('spec')


if __name__ == '__main__':
    write_tfrecords()
