from sklearn.preprocessing import MinMaxScaler

from pipepy.core import Pipeline
from pipepy.engineer import MapColumnPipe

TRANSACTIONS_TRAIN = 'data/transactions-value/train.csv'
TRANSACTIONS_TEST = 'data/transactions-value/test.csv'


def fe_pipeline():
    return Pipeline([
        MapColumnPipe(lambda col: MinMaxScaler().fit_transform(col)),
        #lambda data: umap.UMAP(n_components=2, metric='correlation', verbose=1).fit_transform(data),
    ])


if __name__ == '__main__':
    # data = ddf.read_csv(TRANSACTIONS_TRAIN)
    # data = data.map_partitions(lambda part: part.to_sparse(fill_value=0))
    # data = data.compute().reset_index(drop=True)
    #data = pd.read_csv(TRANSACTIONS_TRAIN)

    #print(data.density)

    import scipy
    import pandas as pd
    import numpy as np
    import seaborn as sns
    sns.set(style="white")

    import h5py

    hf = h5py.File('data/transactions-value/train_red95_601f.h5', 'r')
    X = np.array(hf.get('transactions-value'))
    hf.close()

    X = pd.DataFrame(X)
    print(X.shape)

    pipeline = fe_pipeline()
    X = fe_pipeline().flush(X)

    y = pd.read_csv(TRANSACTIONS_TRAIN)['target']
    y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))  # normalize output to avoid Inf
    print(y.shape)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    model = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=123)

    print('RMSE: ' + str(scipy.sqrt(abs(
        cross_val_score(model, X, y.ravel(), cv=5, scoring='neg_mean_squared_log_error', verbose=2).mean()))))