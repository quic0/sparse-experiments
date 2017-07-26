import pandas as pd
import numpy as np
import os

from ..utils.data_utils import get_file, _extract_archive


def load_data():
    path = get_file('adult.data',
                    origin='https://archive.ics.uci.edu/ml/machine-learning' +
                    '-databases/adult/adult.data',
                    extract=False)

    path_test = get_file('adult.test',
                         origin='https://archive.ics.uci.edu/ml/machine-' +
                         'learning-databases/adult/adult.test',
                         extract=False)

    data = pd.read_csv(path, delimiter=",", header=None,
                       na_values="?",
                       skipinitialspace=True,
                       converters={14: lambda x: int(x == '>50K')})

    data_test = pd.read_csv(path_test, delimiter=",", header=None,
                            skiprows=1, na_values="?",
                            skipinitialspace=True,
                            converters={14: lambda x: int(x == '>50K.')})

    data = pd.concat((data, data_test))

    # drop NaNs
    data = data.dropna()

    # labels are in last column
    labels = data[14].values

    # convert categorial columns to dummy vars
    # column 3 is dropped due to double info
    categorical = [1, 5, 6, 7, 8, 9, 13]
    numerical = [0, 2, 4, 10, 11, 12]

    data = pd.concat((
        data[numerical],
        pd.get_dummies(data[categorical]
                       )
                      ), axis=1).values

    return (data, labels)

    #
    # labels = data[:, -1]
    # data = data[:, :-1]
    #
    # data_missing = np.isnan(data)
    #
    # data = np.concatenate((data, data_missing), axis=1)
    # data = np.where(np.isnan(data), 0, data)
    #
    # return (data, labels)
