# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:36:37 2019

@author: DURSUN
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999 , inplace=True)
df.drop(['id'], 1, inplace=True)



# x is features
X = np.array(df.drop(['class'],1)) 
# y is claster
y = np.array(df['class'])

corr_matrix = df.corr()
print(corr_matrix["class"].sort_values(ascending=False))



df.hist(bins=50, figsize=(12,12))
plt.show()
 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf= svm.SVC()
#clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

predic = clf.predict(X_test)
print(classification_report(y_test,predic))


example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)


