import matplotlib.pyplot as plt
import numpy as np


def plot_decision_tree_feature_importances(model, features):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


# TRY-OUT predicting a feature
#
# def predict_age(data):
#     age_data = data[data.Age.notnull() & data.Pclass.isin([1, 2])]
#     age_data = age_data.dropna()
#     print(age_data.shape)
#     print(data[data.Age.isnull() & data.Pclass.isin([1, 2])].shape)
#
#     features, labels = age_data.drop('Age', axis='columns'), age_data['Age']
#     model = GradientBoostingRegressor(n_estimators=10, max_depth=3)
#     scores = cross_val_score(model, features, labels, cv=10)
#     print(scores.mean())
#     age_model = model.fit(features, labels)
#
#     #plot_feature_importances(age_model, features)
