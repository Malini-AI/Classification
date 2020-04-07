#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import model_selection

#Import all the datasets
train_features=pd.read_csv('train_features.csv',header=None)
train_labels=pd.read_csv('train_label.csv')
test_features=pd.read_csv('test_features.csv',header=None)

train_feature_split, test_feature_split, train_label_split, test_label_split = train_test_split(train_features, train_labels, test_size=0.2)

#Create RandomForestclassifier and fit train_features and tain_label
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
model = RandomForestClassifier(n_estimators=400,random_state=42,bootstrap=False,max_features=30
                               ,max_depth=None,min_samples_leaf=2,oob_score=False)
#model = AdaBoostClassifier(base_estimator=base_model,n_estimators=base_model.n_estimators)
#model = tree.DecisionTreeClassifier(random_state=42,max_depth=8)
model.fit(train_feature_split, train_label_split)
print(model)
prediction = model.predict(test_feature_split)
accuracy = accuracy_score(test_label_split['label'].values,prediction[:,1])
f1 = f1_score(test_label_split['label'].values,prediction[:,1])

#print(prediction)
print(accuracy)
print(f1)

#Predict test labels for test_features.csv
test_labels = model.predict(test_features)

#Extract the labels only(last column)
a = []
for i in range(len(test_labels)):
    temp = int(test_labels[i][1])
    a.append(temp)

#Update submission file with the labels predicted
file=pd.read_csv('sample_Submission.csv')
file['label'] = a
file.to_csv('submission.csv',header=True,index=False)