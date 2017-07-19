import numpy as np

from ..utils.data_utils import get_file


def load_data(path='covtype.data.gz', targetLabel=1):
    path = get_file(path,
                    origin='https://archive.ics.uci.edu/ml/machine-learning-' +
                    'databases/covtype/covtype.data.gz',
                    extract=True)

    path = path[:-3]

    # In this dataset the class label is in column 54 (0-indexed)
    classColumnIndex = 54

    data = np.genfromtxt(path, delimiter=",")

    labels = np.where(data[:, classColumnIndex] == targetLabel, 1, 0)
    features = data[:, :classColumnIndex]

    return (features, labels)
