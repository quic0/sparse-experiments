import numpy as np
import h5py
import os

from ..utils.data_utils import get_file


def load_data(path='BOW2_full.mat'):
    path = os.path.expanduser(os.path.join('~', '.sparse', 'datasets', path))
    if os.path.exists(path):
        f = h5py.File(path)

        data = np.array(f['data']).transpose()
        labels = np.squeeze(np.array(f['classLabels']))
        data = np.where(data > 0, 1, 0)

        return (data, labels)
    else:
        print('Download manually from: https://drive.google.com/uc?' +
              'id=0B409u1yNB1l3YlJMNXU3S3hOTms&export=download')
        print('Save file as: %s' % path)

        # Can't use get_file due to Google auth
        # path = get_file('BOW2_full.mat', origin='https://drive.google.com/uc?' +
        #                 'id=0B409u1yNB1l3YlJMNXU3S3hOTms&export=download',
        #                 extract=False)

        return (None, None)
