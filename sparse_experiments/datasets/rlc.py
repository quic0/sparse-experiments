import numpy as np
import zipfile
import os

from ..utils.data_utils import get_file, _extract_archive


def load_data(path='donation.zip', targetLabel=1):
    path = get_file(path,
                    origin='https://archive.ics.uci.edu/ml/machine-learning' +
                    '-databases/00210/donation.zip',
                    extract=True)

    dirname = os.path.dirname(os.path.realpath(path))
    rlcDir = os.path.join(dirname, 'rlc')
    complete_path = os.path.join(rlcDir, 'rlc.csv')
    if not os.path.exists(complete_path):
        if not os.path.exists(rlcDir):
            os.mkdir(rlcDir)

        f = open(complete_path, 'w')
        for i in range(1, 11):
            _extract_archive(os.path.join(dirname, 'block_%d.zip' % i),
                             path=rlcDir)
            with open(os.path.join(rlcDir, 'block_%d.csv' % i), 'r') as g:
                header = False
                for line in g:
                    if header:
                        header = True
                    else:
                        f.write(line)
        f.close()

    data = np.genfromtxt(complete_path, delimiter=",",
                         skip_header=1, missing_values='?',
                         converters={11: lambda s: float(s == 'TRUE')},
                         dtype=np.float)

    labels = data[:, -1]
    data = data[:, :-1]

    data_missing = np.isnan(data)

    data = np.concatenate((data, data_missing), axis=1)
    data = np.where(np.isnan(data), 0, data)

    return (data, labels)
