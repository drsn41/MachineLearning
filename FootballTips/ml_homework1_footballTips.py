# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:36:37 2019

@author: DURSUN
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('football-tips.csv')
df.drop(['KOD'], 1, inplace=True)
df.dropna(inplace=True)

# x is features
X = np.array(df.drop(['SONUC'],1)) 
# y is claster
y = np.array(df['SONUC'])

#
corr_matrix = df.corr()
print(corr_matrix["SONUC"].sort_values(ascending=False))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=1)

#clf = LogisticRegression()
clf= neighbors.KNeighborsClassifier(n_neighbors= 101)
#clf= svm.SVC()
clf.fit(X_train,y_train)
#
#predictions = clf.predict(X_test)
#print(predictions)
accuracy = clf.score(X_test, y_test)
print(accuracy)
#classification_rep = classification_report(y_test, predictions)
#print(classification_rep)
#conf_mat = confusion_matrix(y_test, predictions)
#print(conf_mat)
example_measures = np.array([  [2.0, 3.0, 2.75] ,  [1.7, 3.3, 3.5]  ])
example_measures = example_measures.reshape(len(example_measures),-1)
#
prediction = clf.predict(example_measures)
print(prediction)

#param_grid = [
#{'n_neighbors': [3, 10, 30], 'p': [2, 4, 6, 8]},
#{'leaf_size': [3, 10]},
#]
#knn = neighbors.KNeighborsClassifier() 
#
#grid_search = GridSearchCV(knn, param_grid, cv=5,
#scoring='neg_mean_squared_error')
#grid_search.fit(X_train, y_train)
#grid_search.best_params_