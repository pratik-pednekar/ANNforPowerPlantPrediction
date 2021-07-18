#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:07:13 2021

@author: pratik
"""

# Importing modules and libraries
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# Importing data (Data from https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
data=pd.read_csv('./Dataset/Folds5x2_pp.csv')
print(data.head())

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

# Splitting to test and train 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Modeling using ANN
## Initializing the Neural Network
ann=tf.keras.models.Sequential() #Sequence of layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

## Building and training the Neural Network
ann.compile(optimizer='adam',loss='mean_squared_error')
ann.fit(X_train,y_train,batch_size=32,epochs=100)

## Anazlyging the results of the NN
y_pred=ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

print("Out of sample r2 score:",r2_score(y_test,y_pred))
print("Out of samplel MSE",mean_squared_error(y_test,y_pred))

print("In of samplel MSE",mean_squared_error(y_train,ann.predict(X_train)))