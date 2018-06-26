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


# from keras.models import Sequential
# from keras.layers import Dense, Dropout
#
# model = Sequential()
# model.add(Dense(X.shape[1], input_shape=(X.shape[1],), kernel_initializer='normal', activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='relu'))
#
# model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop', metrics=['mean_squared_logarithmic_error'])
#
# history = model.fit(X, y, epochs=250, batch_size=75, verbose=1, validation_split=0.3)
#
# plt.plot(history.history['mean_squared_logarithmic_error'])
# plt.plot(history.history['val_mean_squared_logarithmic_error'])
# plt.title('mean_squared_logarithmic_error')
# plt.ylabel('error')
# plt.xlabel('epoch')
# plt.ylim(ymax=0.2, ymin=0)
# plt.xlim(xmin=20)
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
