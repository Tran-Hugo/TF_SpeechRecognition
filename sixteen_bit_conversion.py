import librosa
import librosa.display
import numpy as np
import soundfile as sf

import os
import sys


def convert(file_path: str):
    """convert audio file to 16bits format"""

    y, sr = librosa.load(file_path, sr=16000)

    y_16bits = np.int16(y * 32767)

    sf.write(file_path, y_16bits, sr, format='WAV', subtype='PCM_16')

if __name__ == "__main__":
    try:
        convert(sys.argv[1])
    except IndexError as err:
        print("This module needs a file path as first parameter")
        raise err