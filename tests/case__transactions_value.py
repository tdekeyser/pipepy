import dask.dataframe as ddf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from pipepy.clean import DropColumnPipe
from pipepy.core import Pipeline
from pipepy.engineer import MapColumnPipe
from pipepy.explore import PeekPipe

TRANSACTIONS_TRAIN = 'data/transactions-value/train.csv'
TRANSACTIONS_TRAIN_INCPCA_700 = 'data/transactions-value/train_incPCA_700.h5'
TRANSACTIONS_TEST = 'data/transactions-value/test.csv'


def fit_model(data, model):
    model = model.fit(data)

    plt.title('Eigenvalue analysis')
    plt.plot(model.explained_variance_ratio_.cumsum())
    plt.show()

    return model.transform(data)


def plot_embedding(data):
    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.show()


def fe_pipeline():
    return Pipeline([

        PeekPipe(lambda d: print(d.columns[:10])),

        DropColumnPipe(['ID', 'target']),

        #MapColumnPipe(lambda col: MinMaxScaler().fit_transform(col.reshape(-1, 1))),

        # lambda data: umap.UMAP(n_components=2, metric='correlation', verbose=1).fit_transform(data),

        lambda data: fit_model(data, IncrementalPCA(n_components=700)),  # 98% variance
        #plot_embedding,
    ])


if __name__ == '__main__':
    # data = ddf.read_csv(TRANSACTIONS_TRAIN)
    # data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
    # data = data.compute().reset_index(drop=True)
    data = pd.read_csv(TRANSACTIONS_TRAIN)

    #print(data.density)

    # pipeline = fe_pipeline()
    # reduced_data = pipeline.flush(data)
    # reduced_data.to_hdf(TRANSACTIONS_TRAIN_INCPCA_700, 'TRANSACTIONS_TRAIN_INCPCA_700')
    print('getting train/test')

    X = pd.read_hdf(TRANSACTIONS_TRAIN_INCPCA_700, 'TRANSACTIONS_TRAIN_INCPCA_700')
    X = X.to_sparse(fill_value=0)
    y = data['target']

    print('building model...')

    # TRAIN AND EVALUATE MODEL
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, min_samples_leaf=4, random_state=123)
    print('RMSE: ' + str(
        scipy.sqrt(abs(cross_val_score(model, X, y, cv=2, scoring='neg_mean_squared_log_error').mean()))))
