from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import pickle
import h5py
import os

if not os.path.exists('Flower_17/transfer_learning_VGG16/logistic_regression'):
    os.mkdir('Flower_17/transfer_learning_VGG16/logistic_regression')


db = h5py.File('Flower_17/transfer_learning_VGG16/features.hdf5', 'r')
i = int(db['labels'].shape[0] * 0.75)

X_train = db['features'][:i]
y_train = db['labels'][:i]
X_test = db['features'][i:]
y_test = db['labels'][i:]

params = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(solver = 'lbfgs', multi_class = 'auto'), params, cv=3, n_jobs=-1)
print('... Tuning Hyperparameters')
model.fit(X_train, y_train)

print('Model\'s best hyperparameter: {}'.format(model.best_params_))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names = db['label_names']))

report = classification_report(y_test, y_pred, output_dict = True)
df = pd.DataFrame(report).transpose()
df.to_csv('Flower_17/transfer_learning_VGG16/logistic_regression/classification_report.csv', index = False)

f = open('Flower_17/transfer_learning_VGG16/logistic_regression/logistic_regression_model.cpickle', 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close

db.close()





