import matplotlib.pyplot as plt
import pandas as pd
import scipy
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, Imputer

from pipepy.clean import CategoryToNumericPipe, DropColumnPipe, RemoveOutliersPipe
from pipepy.core import Pipeline
from pipepy.engineer import MapColumnPipe, AddColumnPipe
from pipepy.explore import PeekPipe
from tests.case__utils import plot_decision_tree_feature_importances

HOUSE_TRAIN = 'data/house-prices/train.csv'
HOUSE_TEST = 'data/house-prices/test.csv'


def inspect_features(data, cols):
    for col in cols:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        fig = sns.boxplot(y=data[col])
        fig.set_title('')
        fig.set_ylabel(col)

        plt.subplot(1, 2, 2)
        fig = sns.distplot(data[col].dropna())
        fig.set_ylabel('Number of houses')
        fig.set_xlabel(col)

        plt.show()


def boxcox(price):
    transform, param = scipy.stats.boxcox(price)
    return transform


def fill_garage_blt(data):
    data['GarageYrBlt'] = np.where(data['GarageYrBlt'].isnull(), data['YearBuilt'], data['GarageYrBlt'])
    return data


def train_pipeline():
    return Pipeline([
        PeekPipe(lambda data: print(data.shape)),

        # FEATURE ENGINEERING
        AddColumnPipe(lambda data: data['GarageArea']
                                   + data['GrLivArea']
                                   + data['1stFlrSF']
                                   + data['2ndFlrSF']
                                   + data['TotalBsmtSF'], 'ExtraArea'),
        fill_garage_blt,

        RemoveOutliersPipe(['LotArea']),

        #MapColumnPipe(lambda col: boxcox(col), columns=['LotArea']),
        #PeekPipe(lambda data: inspect_features(data, ['BsmtFinSF1', 'YearBuilt'])),

        DropColumnPipe([
            'Id',
            'MSSubClass',
            'ExterCond', 'Exterior2nd',
            'Utilities',
            'GarageCond', 'GarageQual', 'GarageType',
            'RoofMatl', 'RoofStyle',
            'Heating', 'HeatingQC',
            'Street',
            'MiscFeature',
            'MiscVal',
            'BsmtFinType2', 'BsmtHalfBath', 'BsmtFullBath', 'FullBath', 'HalfBath',
            'Fence',
            'SaleType',
            'Alley',
            'LandSlope', 'LandContour',
            'Electrical',
            '3SsnPorch', 'EnclosedPorch',
            'PavedDrive',
            'Condition2',
            'LowQualFinSF',
            'Foundation',
            'TotRmsAbvGrd',
            'PoolQC',
            'BedroomAbvGr',
            'LotConfig',
        ]),

        MapColumnPipe(lambda col: col.fillna(-1),
                      columns=[
                          'LotFrontage',
                          'MasVnrType',
                          'MasVnrArea',
                          'BsmtExposure',
                          'BsmtFinType1',
                          'FireplaceQu',
                      ]),

        PeekPipe(lambda data: print(data.dtypes)),

        CategoryToNumericPipe(['SalePrice', 'LotArea', 'LotFrontage'], excludes=True),

        #PeekPipe(lambda data: inspect_features(data, ['LotArea'])),

        # PeekPipe(lambda data: print(data.isnull().sum()[data.isnull().sum() > 0])),
        MapColumnPipe(lambda col: Imputer(missing_values='NaN',
                                          strategy='mean',
                                          axis=0).fit_transform(col.values.reshape(-1, 1))),

        # NORMALIZE
        DropColumnPipe(['SalePrice']),
        MapColumnPipe(lambda col: MinMaxScaler().fit_transform(col.values.reshape(-1, 1))),

        PeekPipe(lambda data: print(data.shape))
    ])


if __name__ == '__main__':
    data = pd.read_csv(HOUSE_TRAIN)

    pipeline = train_pipeline()
    data = pipeline.flush(data)

    X, y = data, pipeline.pipes[-3].residue[0]

    # TRAIN AND EVALUATE MODEL
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=123)
    print('RMSE: ' + str(
        scipy.sqrt(abs(cross_val_score(model, X, y, cv=2, scoring='neg_mean_squared_log_error').mean()))))
    model = model.fit(X, y)

    #plot_decision_tree_feature_importances(model, X)

    # MAKE PREDICTIONS AND EXPORT
    test_data = pd.read_csv(HOUSE_TEST)

    id = test_data['Id']
    test_data['SalePrice'] = y

    del pipeline.pipes[3]
    test_data = pipeline.flush(test_data)

    test_predictions = model.predict(test_data)

    submission = pd.DataFrame({'Id': id, 'SalePrice': test_predictions})
    submission.to_csv('data/house-prices/submission4.csv', index=False)
