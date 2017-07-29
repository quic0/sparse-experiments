import numpy as np
import os

from ..utils.data_utils import get_file


def load_data(path='ringnorm.data'):
    path = os.path.expanduser(os.path.join('~', '.sparse', 'datasets', path))
    if os.path.exists(path):
        data = np.genfromtxt(path, delimiter=",", dtype=np.float)

        # labels are in last column
        labels = data[:, -1]

        data = data[:, :-1]

        return (data, labels)
    else:
        print('Download manually from: https://drive.google.com/uc?' +
              'id=0B409u1yNB1l3TW04TDV2YXExaTQ&export=download')
        print('Save file as: %s' % path)

        # Can't use get_file due to Google auth
        # path = get_file('ringnorm.data', origin='https://drive.google.com/uc?' +
        #                 'id=0B409u1yNB1l3TW04TDV2YXExaTQ&export=download',
        #                 extract=False)

        return (None, None)
