"""
Using pipepy to create a classifier for the Titanic dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer, MinMaxScaler

from pipepy.core import Pipeline
from pipepy.pandas_pipe import DropColumnPipe, CategoryToNumericPipe, MapColumnPipe, PeekPipe, AddColumnPipe

TITANIC = 'data/titanic/train.csv'


def parse_titles(names):
    titles = names.str.replace(r'^.+, (.+?[.]) .+$', lambda match: match.group(1))
    titles = titles.apply(lambda t: 'Miss.' if t in ['Mrs.', 'Miss.', 'Mlle.', 'Mme', 'Ms.'] else t)
    # titles = titles.apply(lambda t: 'Special' if t not in ['Miss.', 'Mr.'] else t)
    return titles


def build_pipeline():
    return Pipeline([

        # Feature engineer titles from names
        AddColumnPipe([parse_titles(data.Name)], ['Title']),

        # Turn categorical variables into numeric
        CategoryToNumericPipe(['Embarked', 'Sex', 'Pclass', 'Title']),

        # Deal with missing data
        PeekPipe(lambda data: print(data.isnull().sum())),
        MapColumnPipe(
            lambda age: Imputer(missing_values='NaN',
                                strategy='median',
                                axis=0)
                .fit_transform(age.values.reshape(-1, 1)),
            columns=['Age']
        ),

        # Drop columns that do not affect the model
        DropColumnPipe(['PassengerId', 'Name', 'Cabin', 'Ticket']),
        lambda data: data.dropna(),

        # Normalize
        MapColumnPipe(lambda col: MinMaxScaler()
                      .fit_transform(col.values.reshape(-1, 1)))
    ])


def plot_feature_importances(model, features):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv(TITANIC, na_values="", keep_default_na=False)

    ## CLEANING + FEATURE ENGINEERING
    pipeline = build_pipeline()
    data = pipeline.flush(data)
    print(data.head(1))

    age_data = Pipeline([DropColumnPipe(['Survived', 'Sex', 'Embarked', 'SibSp'])]).flush(data[data.Age.notnull()])

    print(age_data.columns)
    print(age_data.shape)
    features, labels = age_data.drop('Age', axis='columns'), age_data['Age']
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=np.random.RandomState(1))
    scores = cross_val_score(model, features, labels, cv=3)
    print(scores.mean())
    age_model = model.fit(features, labels)

    plot_feature_importances(age_model, features)

    ## TRAIN AND EVALUATE CLASSIFIER
    features, labels = data.drop('Survived', axis='columns'), data['Survived']
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=np.random.RandomState(1))
    scores = cross_val_score(model, features, labels, cv=3)
    print(scores.mean())
    titanic_model = model.fit(features, labels)

    plot_feature_importances(titanic_model, features)
