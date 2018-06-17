"""
Using pipepy to create a classifier for the Titanic dataset.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from pipepy.core import PipeLine
from pipepy.pandas_pipe import DropColumnPipe, CategoryToNumericPipe

TITANIC = 'data/titanic_train.csv'


def build_pipeline():
    return PipeLine([
        DropColumnPipe([
            'PassengerId',
            'Name',
            'Cabin',
            'Ticket'
        ]),
        CategoryToNumericPipe([
            'Embarked',
            'Sex',
            'Pclass'
        ]),
        lambda data: data.fillna(0)
    ])


def split(dataset):
    label_name = 'Survived'
    train, test = train_test_split(dataset, test_size=0.3)
    x_train, x_test = train.drop(label_name, 1), test.drop(label_name, 1)
    y_train, y_test = train[label_name], test[label_name]
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    data = pd.read_csv(TITANIC, na_values="", keep_default_na=False)
    print(data.columns)

    ## CLEANING + FEATURE ENGINEERING
    data = build_pipeline().flush(data)
    print(data)

    ## TRAIN/TEST SPLIT
    x_train, x_test, y_train, y_test = split(data)
    print("Split data: train ({0}, {1}) - test ({2}, {3})"
          .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    ## EVALUATE CLASSIFIER
    model = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=np.random.RandomState(1)) \
        .fit(x_train, y_train)
    print(model.score(x_test, y_test))
