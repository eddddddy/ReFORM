import os
import pickle
import multiprocessing

import tensorflow as tf

import constants
import generate


def get_nth_serialized_train_filepath(n):
    return os.path.join(os.path.dirname(__file__), 'spectrogram_data', f'train{n}.tfrecord')


def to_feature_list(array_2d):
    float_lists = [tf.train.Feature(float_list=tf.train.FloatList(value=row)) for row in array_2d]
    return tf.train.FeatureList(feature=float_lists)


def serialize(array):
    feature_list = {'spec': to_feature_list(array)}
    example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_list))

    return example.SerializeToString()


def write_tfrecord(n):
    raw_filepath = generate.get_nth_train_filepath(n)
    serialized_filepath = get_nth_serialized_train_filepath(n)
    with open(raw_filepath, 'rb') as raw_file:
        with tf.io.TFRecordWriter(serialized_filepath) as writer:
            try:
                while True:
                    data = pickle.load(raw_file)
                    writer.write(serialize(data))
            except EOFError:
                pass


def write_all_tfrecords():
    workers = []
    for i in range(generate.NUM_THREADS):
        workers.append(multiprocessing.Process(target=write_tfrecord, args=(i,)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


feature_description = {
    'spec': tf.io.FixedLenSequenceFeature(constants.NUM_BINS, tf.float32)
}


def decode(example):
    return tf.io.parse_single_sequence_example(example, sequence_features=feature_description)[1].get('spec')


if __name__ == '__main__':
    write_all_tfrecords()
