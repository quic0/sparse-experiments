from ..utils.data_utils import get_file


def load_data(path='covtype.data.gz', targetLabel=1):
    path = get_file(path,
                    origin='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                    extract=True)

    path = path[:-3]
