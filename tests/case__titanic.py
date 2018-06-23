"""
Using pipepy to create a classifier for the Titanic dataset.

Two extra features are engineered: titles and a family count. In order
to deal with missing data, median values are imputed. Next, the Age feature
was grouped into 3 categories.
The pipeline eventually gets the following features to be used in the model:
Pclass  Sex  Age  Fare  Embarked  Title  FamilyCount

Modelling method: stacked ensemble
- train a set of base learners
- predict the training set using cross-validation to avoid fitting the target
- add the base learner predictions as features
- train a top learner on the new dataset

Best accuracy as of June 2018 on the Kaggle test set is 81.339%.

:author: Tom De Keyser
"""
from typing import Sequence, TypeVar

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.svm import SVC

from pipepy.clean import DropColumnPipe, CategoryToNumericPipe, VariableToBinPipe
from pipepy.core import Pipeline
from pipepy.engineer import AddColumnPipe, MapColumnPipe

Learner_t = TypeVar('Learner_t')

TITANIC_TRAIN = 'data/titanic/train.csv'
TITANIC_TEST = 'data/titanic/test.csv'


def parse_titles(names):
    titles = names.str.replace(r'^.+, (.+?[.]) .+$', lambda match: match.group(1))
    titles = titles.apply(lambda t: 'Miss.' if t in ['Mrs.', 'Miss.', 'Mme.', 'Ms.'] else t)
    titles = titles.apply(lambda t: 'Special' if t not in ['Miss.', 'Mr.', 'Master.', 'Rev.', 'Dr.'] else t)
    return titles


def cleaning_and_feature_pipeline():
    return Pipeline([
        # Feature engineer new columns
        AddColumnPipe(lambda data: parse_titles(data.Name), 'Title'),
        AddColumnPipe(lambda data: data.Parch + data.SibSp, 'FamilyCount'),

        # Drop columns that do not affect the model
        DropColumnPipe(['PassengerId', 'Name', 'Cabin', 'Parch', 'SibSp', 'Ticket']),

        # Turn categorical variables into numeric
        CategoryToNumericPipe(['Title', 'Sex', 'Embarked']),

        # Deal with missing data
        MapColumnPipe(
            lambda col: Imputer(missing_values='NaN',
                                strategy='median',
                                axis=0).fit_transform(col.values.reshape(-1, 1)),
            columns=['Age', 'Fare', 'Embarked']),

        # Group Age into 3 categories
        VariableToBinPipe(bins=3, columns=['Age']),

        # Normalize
        MapColumnPipe(lambda col: MinMaxScaler().fit_transform(col.values.reshape(-1, 1)))
    ])


def stacked_ensemble_train(X, y,
                           base_learners: Sequence[Learner_t] = None,
                           top_learner: Learner_t = None,
                           cross_val=6) -> (Sequence[Learner_t], Learner_t):
    base_predictions = [cross_val_predict(learner, X, y, cv=cross_val) for learner in base_learners]
    base_learners = [learner.fit(X, y) for learner in base_learners]

    for i, prediction in enumerate(base_predictions):
        X['base' + str(i)] = prediction

    print('Cross-validation accuracy of top learner: %f' % cross_val_score(top_learner, X, y, cv=10).mean())

    return base_learners, top_learner.fit(X, y)


def stacked_ensemble_predict(X,
                             base_learners: Sequence[Learner_t] = None,
                             top_learner: Learner_t = None) -> Sequence[float]:
    base_predictions = [learner.predict(X) for learner in base_learners]

    for i, prediction in enumerate(base_predictions):
        X['base' + str(i)] = prediction

    return top_learner.predict(X)


if __name__ == "__main__":
    data = pd.read_csv(TITANIC_TRAIN, na_values="", keep_default_na=False)
    test_data = pd.read_csv(TITANIC_TEST, na_values="", keep_default_na=False)

    # CLEANING + FEATURE ENGINEERING

    pipeline = cleaning_and_feature_pipeline()
    data = pipeline.flush(data)
    print(data.head(1))

    # EVALUATE CLASSIFIER -- STACKED ENSEMBLE

    data = data.sample(frac=1)  # shuffle data
    X, y = data.drop('Survived', axis='columns'), data['Survived']

    base_learners, top_learner = stacked_ensemble_train(
        X, y, cross_val=6,
        base_learners=[
            RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_leaf=4, random_state=2345),
            SVC(kernel='linear', C=0.5),
            ExtraTreesClassifier(n_estimators=20, max_depth=3, min_samples_leaf=4, random_state=678)
        ],
        top_learner=GradientBoostingClassifier(n_estimators=40, max_depth=3, min_samples_leaf=4, random_state=123)
    )

    # MAKE PREDICTIONS AND EXPORT

    test_data = pipeline.flush(test_data)

    test_predictions = stacked_ensemble_predict(test_data,
                                                base_learners=base_learners,
                                                top_learner=top_learner)

    submission = pd.DataFrame({'PassengerId': pipeline.pipes[2].residue[6], 'Survived': test_predictions.astype(int)})
    submission.to_csv('data/titanic/submission2.csv', index=False)
