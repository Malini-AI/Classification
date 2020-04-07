#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
from math import sqrt
import sys
from statistics import mean
from random import randrange
from sklearn import preprocessing
import itertools


# User Input for Dataset, Number of Folds & Number of Neighbors(k)
choice1 = int(input("Enter 1 if you want to enter test and train set seperately or 0 if you want to enter only one file: "))
if(choice1==0):
    file_name = input("Enter file name: ")
    fold = int(input("Enter number of folds: "))
else:
    train_file_name = input("Enter Train file name: ")
    test_file_name = input("Enter Test file name: ")
neighbors = int(input("Enter number of Neighbors to consider: "))
choice2 = int(input("Enter 1 for Normalizing Data or 0 for not Normalizing Data: "))

# Reading file into Dataframe
if choice1==1:
    trainData = pd.read_csv(train_file_name, sep='\t', lineterminator='\n', header=None)
    testData = pd.read_csv(test_file_name, sep='\t', lineterminator='\n', header=None)
else:
    Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)
    Data

# Function to check if the dataset has nominal(string) and/or continuous(numerical) and return the columns containing strings
def check_string(data):
    index=[]
    flag = False
    #print(data.iloc[0])
    for i in range(len(data.iloc[0])):
        #print(data[i][0])
        st = str(data[i][0])
        if(st.replace('.','',1).isdigit() == False):
            flag=True
            index.append(i)
    return flag, index      

# Funtion to convert string to its equivalent one hot enconding value
def one_hot_encoding(data):
    le = preprocessing.LabelEncoder()
    unique_vals = np.unique(data)
    #print(unique_vals)
    le.fit(unique_vals)
    #print(data.shape)
    data = le.transform(data)
    #print(data)
    return data

# Check if the dataset has any string values and converting the string values to its one hot encoding values
if choice1==0:
    flag, str_index = check_string(Data)
    np_data = Data.to_numpy()
    #print(flag, str_index)
    for i in str_index:
        np_data[:,i]=one_hot_encoding(np_data[:,i])
        #np_data[0]
    Data=pd.DataFrame(np_data)
    #Data
else:
    flag, str_index = check_string(trainData)
    np_data = trainData.to_numpy()
    #print(flag, str_index)
    for i in str_index:
        np_data[:,i]=one_hot_encoding(np_data[:,i])
        #np_data[0]
    trainData=pd.DataFrame(np_data)
    #Data
    flag, str_index = check_string(testData)
    np_data = testData.to_numpy()
    #print(flag, str_index)
    for i in str_index:
        np_data[:,i]=one_hot_encoding(np_data[:,i])
        #np_data[0]
    testData=pd.DataFrame(np_data)
    #Data


if choice1==0:
    print(Data)
else:
    print(trainData)
    print(testData)


if(choice2==1):
    #Rescaling the feature values to range between 0 - 1 
    features = Data.iloc[:,:-1]
    normalized = preprocessing.normalize(features)
    normalize = pd.DataFrame(normalized)
    Data = pd.concat([normalize,Data.iloc[:,-1]], axis = 1)
    print(Data)

#Function to calculate Euclidian Distance between 2 Points
def eucl_dist(p1, p2):
    d = 0.0
    for i in range(len(p1)-1):
        d = d + (p1[i] - p2[i])**2
    d = sqrt(d)
    return d


#Function to predict the output class of the test data sample
def predict_class(train, row, neighbors):
    distances = []
    k_neighbors = []
    output = []
    for train_row in train:
        #print(train_row)
        #Computing the distance of the Test sample from all train data samples
        #print("row:",row)
        #print("train row:",train_row)
        dist = eucl_dist(row,train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup: tup[1])
    #Find the k closest samples 
    for i in range(neighbors):
        k_neighbors.append(distances[i][0])
    for k in k_neighbors:
        output.append(k[-1])
    #Find the prediction of the test sample
    final_class = max(set(output), key=output.count)
    return final_class

#Function to execute the KNN algorithm 
def KNN(train,test,neighbors):
    predictions = []
    for row in test:
        output_class = predict_class(train,row,neighbors)
        predictions.append(output_class)
    return predictions

# Function to calculate the performance metrics for the algorithm
def evaluation(predicted_value, actual_value):
    TP=TN=FP=FN=0
    for i in range(len(actual_value)):
        if(actual_value[i]==1 and predicted_value[i]==1):
            TP +=1
        elif(actual_value[i]==0 and predicted_value[i]==0):
            TN += 1
        elif(actual_value[i]==1 and predicted_value[i]==0):
            FN += 1
        elif(actual_value[i]==0 and predicted_value[i]==1):
            FP += 1
    print("TP:",TP,"TN:",TN,"FP:",FP,"FN:",FN)
    if (TP+TN+FP+FN)==0:
        accuracy=0
    else:
        accuracy=(TP+TN)/(TP+TN+FP+FN)
    if (TP+FP)==0:
        precision=0
    else:
        precision=(TP)/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=(TP)/(TP+FN)
    if ((2*TP)+FN+FP)==0:
        f1=0
    else:
        f1=(2*TP)/((2*TP)+FN+FP)
    return accuracy, precision, recall, f1

# Function to create K-Folds of the given dataset into test and train datasets
def create_fold(data,fold):
    #print(len(data))
    
    if fold==1:
        #train=data
        n=(int)(len(data)*0.8)
        #print(n)
        train = data.iloc[:n].values #indexes rows for training data
        test = data.iloc[n:].values
        print(len(train), len(test))
    else:
        train = []
        test_f=[]
        test=[]
        l = int(len(data)/fold)
        for i in range(fold):
            train_fold=data
            test_fold=[]
            for j in range(l):
                index = randrange(len(train_fold))
                #print(index)
                test_fold.append(train_fold.iloc[index])
                train_fold=train_fold.drop(train_fold.index[index])
            test_f.append(test_fold)
            t = train_fold.iloc[:,:].values
            train.append(t)
            #print("fold_train-"+str(i),"---",str(len(train[0][0])))
        # Converting the rows which are a Series into Arrays 
        for i in range(fold):
            arr = []
            for j in range(len(test_f[i])):
                temp = test_f[i][j].values
                arr.append(temp)
            test.append(arr)   
    return test,train

# Create folds for the given datasets
if(choice1==0):
    test_fold,train_fold = create_fold(Data,fold)
else:
    train_fold = trainData.iloc[:,:].values #indexes rows for training data
    test_fold = testData.iloc[:,:].values
    fold=1

# Loop through all Folds, predict output class by KNN algorithm and calculate performance metrics for each fold
accuracy=[]
precision=[]
recall=[]
f1=[]
i=1
if fold==1:
    actual_value = []
    print("Fold Iteration: ",i)
    train=train_fold
    test=test_fold
    print("Train Dataset: ",len(train))
    print("Test Dataset: ",len(test))
    predicted_value = KNN(train,test,neighbors)
    for k in test:
        actual_value.append(k[-1])
    # Compute confusion matrix factors
    a,p,r,f=evaluation(predicted_value, actual_value)
    if(a!=0):
        accuracy.append(a)
    if(p!=0):
        precision.append(p)
    if(r!=0):
        recall.append(r)
    if(f!=0):
        f1.append(f)
else:
    for f in range(fold):
        actual_value = []
        print("Fold Iteration: ",i)
        train=train_fold[f]
        test=test_fold[f]
        print("Train Dataset: ",len(train))
        print("Test Dataset: ",len(test))
        # predict output class for test data
        predicted_value = KNN(train,test,neighbors)
        for k in test:
            actual_value.append(k[-1])
        # Compute performance metrics
        a,p,r,f=evaluation(predicted_value, actual_value)
        if(a!=0):
            accuracy.append(a)
        if(p!=0):
            precision.append(p)
        if(r!=0):
            recall.append(r)
        if(f!=0):
            f1.append(f)
        i+=1
# Calculate and print Average of all folds for each of the performance metrics
print("Accuracy: ", mean(accuracy))
print("Precision: ", mean(precision))
print("Recall: ", mean(recall))
print("F1 measure: ", mean(f1))