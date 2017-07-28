import numpy as np
import os

from ..utils.data_utils import get_file


def load_data(path='checker.dat'):
    path = get_file(path, origin='https://drive.google.com/uc?' +
                    'id=0B409u1yNB1l3cXo4WmRKYmFORzQ&export=download',
                    extract=False)

    data = np.genfromtxt(path, delimiter=",", dtype=np.float)

    # labels are in last column
    labels = data[:, -1]
    
    data = data[:, :-1]

    return (data, labels)
