import numpy as np
import librosa

import constants


def compute_spectrogram(y, sr):
    assert len(y.shape) == 1

    if len(y) < constants.NUM_SAMPLES:
        y = np.pad(y, (0, constants.NUM_SAMPLES - len(y)))

    mel_spec = librosa.feature.melspectrogram(y,
                                              sr=sr,
                                              n_fft=constants.FRAME_SIZE,
                                              hop_length=constants.HOP_SIZE,
                                              n_mels=constants.NUM_BINS,
                                              center=False)
    return librosa.core.power_to_db(mel_spec).transpose()

    
def compute_normal_spectrogram(y, sr):
    """
    Return a mel spectrogram with all values scaled to
      the range [0, 1]
    """
    assert len(y.shape) == 1
    y = np.asfortranarray(y)
    
    if len(y) < constants.NUM_SAMPLES:
        y = np.pad(y, (0, constants.NUM_SAMPLES - len(y)))
        
    mel_spec = librosa.feature.melspectrogram(y,
                                              sr=sr,
                                              n_fft=constants.FRAME_SIZE,
                                              hop_length=constants.HOP_SIZE,
                                              n_mels=constants.NUM_BINS,
                                              fmin=constants.MIN_FREQ,
                                              fmax=constants.MAX_FREQ,
                                              center=False)
    norm = np.log(mel_spec + 1e-15)
    norm -= norm.min()
    norm /= norm.max()
    return norm.transpose()
