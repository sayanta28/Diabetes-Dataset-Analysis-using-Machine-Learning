# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:58:38 2020

@author: Sayanta Chowdhury
"""

import pandas as pd
import numpy as np

#Import Data Set
dataSet = pd.read_csv("diabetes.csv")

#print(dataSet.head())

#Total Null Value Check
print("How many Null value are here: ")
print(dataSet.isnull().sum())

## print(dataSet.isnull().values.any())

#All Column Data Type Check
dataSet.dtypes

# dataSet.corr()
# Diabetes True/False Count
diabetes_true_count = len(dataSet.loc[dataSet['Outcome'] == True])
diabetes_false_count = len(dataSet.loc[dataSet['Outcome'] == False])

print("True Outcome: {0}, False Outcome: {1}".format(diabetes_true_count, diabetes_false_count))

#Missing Zeros without Outcome Column
print("How many Zero value are here: ")
print(dataSet.iloc[:, 0 : 8].eq(0).sum())

# Feature Cloumn or Independent Variable
X = dataSet.iloc[:,:-1].values

#Dependent column or Predict Class
y = dataSet.iloc[:,8].values

# Zero fill with mean of the column
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values= 0, strategy = 'mean')
X[:,1:8] = fill_values.fit_transform(X[:,1:8])

#Trin/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) 

#Decision Tree Classifier Algorithm
from sklearn.tree import DecisionTreeClassifier
DTreeClassifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
DTreeClassifier.fit(X_train, y_train)

DTree_y_pred = DTreeClassifier.predict(X_test)

# Train/test Split Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm_DTree = confusion_matrix(y_test, DTree_y_pred)
ac_DTree = accuracy_score(y_test, DTree_y_pred)
print("Decision Tree Train/test Split Confusion Matrix :")
print(cm_DTree)
print("Decision Tree Train/test Split Accurary: ", ac_DTree)


# K Fold Cross validation of decision Tree
from sklearn.model_selection import cross_val_score
DTscore = cross_val_score(DTreeClassifier, X, y, cv= 11).mean()
print("Accuracy of K fold cross validation using Decision Tree:", DTscore)
#Try using CV = 11 for Max Score


#Startified k Fold

from sklearn.model_selection import StratifiedKFold

accuracy = []
skf = StratifiedKFold(n_splits=11, random_state = None)
skf.get_n_splits(X, y)

#print("Confusion Matrix of K fold Cross validation ")
for train_index, test_index in skf.split(X,y):
   # print("Train: ",train_index, "Validation: ", test_index)
    X1_train, X1_test = X[train_index], X[test_index]
    y1_train, y1_test = y[train_index], y[test_index] 
    
    DTreeClassifier.fit(X1_train, y1_train)
    prediction = DTreeClassifier.predict(X1_test)
    #print( confusion_matrix(y1_test, prediction) )
    score = accuracy_score(y1_test, prediction ) 
    accuracy.append(score)
# print(accuracy)
# numpy
print("Accuracy of Startified K fold cross validation using Decision Tree:", np.array(accuracy).mean())


