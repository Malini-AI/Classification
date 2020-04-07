#!/usr/bin/env python
# coding: utf-8

from csv import reader
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import numpy as np 
import numpy as np
from sklearn import preprocessing
from statistics import mean

# User Input for Dataset & Number of Folds
file_name = input("Enter file name: ")
df = pd.read_csv(file_name, delimiter="\t",header=None)

print("Enter fold = 0 for dataset 4 or demo dataset\n")
fold = int(input("K_fold: "))

if fold==0:
    t=(input("test data: "))
    #store the test data in a file
    f = open("Test.txt", "w")
    f.write(t)
    f.close()

if fold == 0:
    TEST = pd.read_csv('Test.txt', delimiter=",",header=None)
    print(TEST)

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

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(data):
    summaries = [(sum(column)/float(len(column)), sqrt(sum([(x-sum(column)/float(len(column)))**2 for x in column]) / float(len(column)-1)), len(column)) for column in zip(*data)]
    del(summaries[-1])
    return summaries

def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict_class(summarize, row)
        predictions.append(output)
    return(predictions)

# Split dataset by class then calculate statistics for each row
def summarize_by_class(data):
    #separated = separate_by_class(data)
    separated = dict()
    for i in range(len(data)):
        v = data[i]
        c_val = v[-1]
        if (c_val not in separated):
            separated[c_val] = list()
        separated[c_val].append(v)
    summaries = dict()
    for c_val, rows in separated.items():
        summaries[c_val] = summarize_dataset(rows)
    return summaries

# Calculate the probabilities of predicting each class for a given row
def class_probability(summaries, row):
    if fold==0:
        pc=[]
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            print("Label: ",class_value) 
            pc.append(probabilities[class_value])
            print("p[H" + str(class_value) +"]: ",probabilities[class_value])
#             print("pb",probabilities)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= (1 / (sqrt(2 * pi) * stdev)) * exp(-((row[i]-mean)**2 / (2 * stdev**2 )))  
    
        print("\n𝑝(𝑋|𝐻0)𝑝(𝐻0): ",probabilities[0])
        print("𝑝(𝑋|𝐻1)𝑝(𝐻1): ",probabilities[1])
        return probabilities
    else:
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= (1 / (sqrt(2 * pi) * stdev)) * exp(-((row[i]-mean)**2 / (2 * stdev**2 )))
        return probabilities

# Predict the class for a given row
def predict_class(summaries, row):
    probabilities = class_probability(summaries, row)
    best_label, best_prob = -100, -1
    for class_value, probability in probabilities.items():
        if best_label == -100 or probability > best_prob:
            best_prob = probability
            best_label = class_value
    if fold==0:
        print("\n","Predicted label for given test data: ",best_label)
    return best_label

# Function to create K-Folds of the given dataset into test and train datasets
def create_fold(data,fold,df1):   
    if (fold==1 or fold==0):
        if fold==1:
            n=(int)(len(data)*0.8)
            train = data.iloc[:n].values #indexes rows for training data
            test = data.iloc[n:].values
        if fold==0:
            #train=data
            n=(int)(len(data))
            train = data.iloc[:n].values #indexes rows for training data
            test = df1.values       
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


# # Append two test data and train data

# In[8]:


if fold == 0:
    f=df.append(TEST)
    f=pd.DataFrame(f)
    print(f)

# Check if the dataset has any string values and converting the string values to its one hot encoding values
if fold == 0:
    flag, str_index = check_string(f)
    np_data = f.to_numpy()
    
    for i in str_index:
        np_data[:,i]=one_hot_encoding(np_data[:,i])
    df=pd.DataFrame(np_data)
    
    df1=df.iloc[len(df)-1,:4]
    df=df.drop(df.index[-1])
    df1=pd.DataFrame(df1)
    df1=df1.transpose()
else:
    flag, str_index = check_string(df)
    np_data = df.to_numpy()
    for i in str_index:
        np_data[:,i]=one_hot_encoding(np_data[:,i])
    df=pd.DataFrame(np_data)
    #print(df)
        
if fold == 0:
    Test_fold , Train_fold = create_fold(df,fold,df1)
else:
    df1=0
    Test_fold , Train_fold = create_fold(df,fold,df1)       

# Loop through all Folds, create Decision Tree and calculate performance metrics for each fold
# Loop through all Folds, predict output class by KNN algorithm and calculate performance metrics for each fold
accuracy=[]
precision=[]
recall=[]
f1=[]
i=1

if fold==0:
    if(df1.shape[0]==1):
        actual_value = []
        #print("Fold Iteration: ",i)
        train=Train_fold
        test=Test_fold
        print("Train Dataset: ",len(train))
        print("Test Dataset: ",len(test))
        print("\n")
        predicted_value = naive_bayes(train, test)
        #print("Predicted Class: ",predicted_value)
        
if fold==1:
    actual_value = []
    print("Fold Iteration: ",i)
    train=Train_fold
    test=Test_fold
    print("Train Dataset: ",len(train))
    print("Test Dataset: ",len(test))
    print("\n")
    predicted_value = naive_bayes(train, test)
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
    #Calculate and print Average of all folds for each of the performance metrics
    print("\nAccuracy: ", mean(accuracy))
    print("Precision: ", mean(precision))
    print("Recall: ", mean(recall))
    print("F1 measure: ", mean(f1))
    
if fold!=0 and fold!=1:
    for f in range(fold):
        actual_value = []
        print("Fold Iteration: ",i)
        train=Train_fold[f]
        test=Test_fold[f]
        print("Train Dataset: ",len(train))
        print("Test Dataset: ",len(test))
        print("\n")
        # predict output class for test data
        predicted_value = naive_bayes(train, test)
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
        i+=1
    #Calculate and print Average of all folds for each of the performance metrics
    print("\nAccuracy: ", mean(accuracy))
    print("Precision: ", mean(precision))
    print("Recall: ", mean(recall))
    print("F1 measure: ", mean(f1))