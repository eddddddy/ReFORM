import tensorflow as tf
from tensorflow.keras import layers, models

from constants import *


def ConvBN(*args, **kwargs):
    def _ConvBN(model, name):
        model = layers.Conv2D(*args, **kwargs, name=f'conv_{name}')(model)
        model = layers.BatchNormalization(axis=-1, name=f'bn_{name}')(model)
        model = layers.ReLU()(model)
        return model
    return _ConvBN
    
    
'''def ConvBN(*args, **kwargs):
    def _ConvBN(model, name):
        model = layers.Conv1D(*args, **kwargs, name=f'conv_{name}')(model)
        model = layers.BatchNormalization(axis=-1, name=f'bn_{name}')(model)
        model = layers.ReLU()(model)
        return model
    return _ConvBN'''


def ConvTransposeBN(*args, **kwargs):
    def _ConvTransposeBN(model, name):
        model = layers.Conv2DTranspose(*args, **kwargs, name=f'conv_transpose_{name}')(model)
        model = layers.BatchNormalization(axis=-1, name=f'bn_{name}')(model)
        model = layers.ReLU()(model)
        return model
    return _ConvTransposeBN
    

def construct():
    net_input = layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')
    model = layers.Reshape((NUM_FRAMES, NUM_BINS, 1))(net_input)

    conv1 = ConvBN(32, (11, 9), strides=(5, 4), padding='same')(model, name=1)
    conv1_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux1_1')(conv1)
    conv1_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv1_aux')(conv1_aux)

    conv2 = ConvBN(64, (9, 5), strides=(4, 2), padding='same')(conv1, name=2)
    conv2_aux = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(conv2, name='aux2_1')
    conv2_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux2_2')(conv2_aux)
    conv2_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv2_aux')(conv2_aux)

    conv3 = ConvBN(192, (5, 5), strides=(2, 2), padding='same')(conv2, name=3)
    conv3_aux = ConvTransposeBN(64, (3, 3), strides=(2, 2), padding='same')(conv3, name='aux3_1')
    conv3_aux = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(conv3_aux, name='aux3_2')
    conv3_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux3_3')(conv3_aux)
    conv3_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv3_aux')(conv3_aux)

    conv4 = ConvBN(192, (5, 5), strides=(2, 2), padding='same')(conv3, name=4)
    conv4_aux = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(conv4, name='aux4_1')
    conv4_aux = ConvTransposeBN(64, (3, 3), strides=(2, 2), padding='same')(conv4_aux, name='aux4_2')
    conv4_aux = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(conv4_aux, name='aux4_3')
    conv4_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux4_4')(conv4_aux)
    conv4_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv4_aux')(conv4_aux)
        
    conv5 = ConvBN(384, (5, 5), strides=(2, 2), padding='same')(conv4, name=5)
    conv5_aux = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(conv5, name='aux5_1')
    conv5_aux = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(conv5_aux, name='aux5_2')
    conv5_aux = ConvTransposeBN(64, (3, 3), strides=(2, 2), padding='same')(conv5_aux, name='aux5_3')
    conv5_aux = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(conv5_aux, name='aux5_4')
    conv5_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux5_5')(conv5_aux)
    conv5_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv5_aux')(conv5_aux)

    conv6 = ConvBN(384, (5, 5), strides=(2, 2), padding='same')(conv5, name=6)
    conv6_aux = ConvTransposeBN(384, (3, 3), strides=(2, 2), padding='same')(conv6, name='aux6_1')
    conv6_aux = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(conv6_aux, name='aux6_2')
    conv6_aux = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(conv6_aux, name='aux6_3')
    conv6_aux = ConvTransposeBN(64, (3, 3), strides=(2, 2), padding='same')(conv6_aux, name='aux6_4')
    conv6_aux = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(conv6_aux, name='aux6_5')
    conv6_aux = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='aux6_6')(conv6_aux)
    conv6_aux = layers.Reshape((NUM_FRAMES, NUM_BINS), name='conv6_aux')(conv6_aux)

    conv7 = ConvBN(512, (5, 3), strides=(2, 1), padding='same')(conv6, name=7)
    decoder = ConvTransposeBN(384, (3, 3), strides=(2, 1), padding='same')(conv7, name='decoder_1')
    decoder = ConvTransposeBN(384, (3, 3), strides=(2, 2), padding='same')(decoder, name='decoder_2')
    decoder = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(decoder, name='decoder_3')
    decoder = ConvTransposeBN(192, (3, 3), strides=(2, 2), padding='same')(decoder, name='decoder_4')
    decoder = ConvTransposeBN(64, (3, 3), strides=(2, 2), padding='same')(decoder, name='decoder_5')
    decoder = ConvTransposeBN(32, (5, 3), strides=(4, 2), padding='same')(decoder, name='decoder_6')
    decoder = layers.Conv2DTranspose(1, (7, 5), strides=(5, 4), activation='sigmoid', padding='same', name='decoder_7')(decoder)
    decoder = layers.Reshape((NUM_FRAMES, NUM_BINS), name='decoder')(decoder)
        
    net_outputs = [conv1_aux, conv2_aux, conv3_aux, conv4_aux, conv5_aux, conv6_aux, decoder]

    model = models.Model(inputs=net_input, outputs=net_outputs)
    tf.keras.utils.plot_model(model, show_shapes=True)
    return model
    
    
def construct_for_pretraining():
    net_input = layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')
    model = net_input

    conv1 = ConvBN(128, 11, strides=5, padding='same')(model, name='encoder1')
    conv2 = ConvBN(256, 9, strides=4, padding='same')(conv1, name='encoder2')

    aux1 = layers.UpSampling1D(size=4)(conv2)
    aux1 = ConvBN(128, 9, padding='same')(aux1, name='aux1_1')
    aux1 = layers.UpSampling1D(size=5)(aux1)
    aux1 = layers.Conv1D(128, 11, padding='same', activation='sigmoid', name='aux1')(aux1)

    conv3 = ConvBN(256, 5, strides=2, padding='same')(conv2, name='encoder3')
    conv4 = ConvBN(512, 5, strides=2, padding='same')(conv3, name='encoder4')

    aux2 = layers.UpSampling1D(size=2)(conv4)
    aux2 = ConvBN(256, 5, padding='same')(aux2, name='aux2_1')
    aux2 = layers.UpSampling1D(size=2)(aux2)
    aux2 = ConvBN(256, 5, padding='same')(aux2, name='aux2_2')
    aux2 = layers.UpSampling1D(size=4)(aux2)
    aux2 = ConvBN(128, 9, padding='same')(aux2, name='aux2_3')
    aux2 = layers.UpSampling1D(size=5)(aux2)
    aux2 = layers.Conv1D(128, 11, padding='same', activation='sigmoid', name='aux2')(aux2)

    conv5 = ConvBN(512, 5, strides=2, padding='same')(conv4, name='encoder5')
    conv6 = ConvBN(1024, 5, strides=2, padding='same')(conv5, name='encoder6')
    conv7 = ConvBN(1024, 5, strides=2, padding='same')(conv6, name='encoder7')

    decoder = layers.UpSampling1D(size=2)(conv7)
    decoder = ConvBN(1024, 5, padding='same')(decoder, name='decoder1')
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = ConvBN(512, 5, padding='same')(decoder, name='decoder2')
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = ConvBN(512, 5, padding='same')(decoder, name='decoder3')
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = ConvBN(256, 5, padding='same')(decoder, name='decoder4')
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = ConvBN(256, 5, padding='same')(decoder, name='decoder5')
    decoder = layers.UpSampling1D(size=4)(decoder)
    decoder = ConvBN(128, 9, padding='same')(decoder, name='decoder6')
    decoder = layers.UpSampling1D(size=5)(decoder)
    decoder = layers.Conv1D(128, 11, padding='same', activation='sigmoid', name='decoder')(decoder)

    net_outputs = [aux1, aux2, decoder]

    model = models.Model(inputs=net_input, outputs=net_outputs)
    return model
    
    
def reshape_last_layer(model):
    model = layers.Reshape((2048,), name='embedded')(model)
    return model


def get_model():
    net_input = layers.Input(shape=(NUM_FRAMES, NUM_BINS), name='input')
    model = construct(net_input)
    #model = reshape_last_layer(model)
    model = models.Model(inputs=net_input, outputs=model)

    model.load_weights('checkpoints/weights-all.hdf5', by_name=True)
    return model
