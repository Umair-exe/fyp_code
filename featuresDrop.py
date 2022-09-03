# report which features were selected by RFE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

data = pd.read_csv('encoded_file.csv');

X = data.drop('Class', axis=1)
target = data['Class']

rfc = RandomForestClassifier(random_state=0)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)


print('Optimal number of features: {}'.format(rfecv.n_features_))

print(np.where(rfecv.support_ == False)[0])

X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)