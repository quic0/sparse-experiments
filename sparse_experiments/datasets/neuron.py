import numpy as np
import h5py

from ..utils.data_utils import get_file


def load_data():
    path = get_file('neurofinder.02.00_10_237_312_31x31.h5',
                    origin='https://dl.dropboxusercontent.com/s/' +
                    'a8uffxheku6i78h/neurofinder.02.00_10_237_312_31x31.h5',
                    extract=False)

    # load and reshape data
    f = h5py.File(path)
    data = np.array(f['data']).transpose().reshape((31 ** 2, -1))

    # no labels available
    return (data, None)
