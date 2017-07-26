import numpy as np
import zipfile
import os

from ..utils.data_utils import get_file, _extract_archive


def load_data(path='letter-recognition.data', targetLabel=1):
    path = get_file(path,
                    origin='https://archive.ics.uci.edu/ml/machine-learning' +
                    '-databases/letter-recognition/letter-recognition.data',
                    extract=False)

    # converter converts letters to numbers in 0-25
    data = np.genfromtxt(path, delimiter=",", dtype=np.float,
                         converters={0: lambda x: float(ord(x) - 65)})

    # Letter O is the positive class
    labels = np.where(data[:, 0] == 14, 1, 0)
    data = data[:, 1:]

    return (data, labels)
