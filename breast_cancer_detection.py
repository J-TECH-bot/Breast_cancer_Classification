# -*- coding: utf-8 -*-
"""Breast_Cancer_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IRAT9aEjL31H54EDIwZ6x51qq3eWerWq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')

df

# M(Malignant) = 1
# B(Benign) = 0
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

df.tail()

df.describe()

df.info()

df.drop(['Unnamed: 32'], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)

df.head()

df.isnull().sum()

df.duplicated().sum()

# GroupBy according to diagnosis
df.groupby('diagnosis').mean()

#Spliting Dataframe
X = df.drop(['diagnosis'], axis=1)
Y = df['diagnosis']

X

Y

#Splitting data for trainning and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

model.fit(X_train, Y_train)

model.predict(X_test)

acc = accuracy_score(Y_test, model.predict(X_test))

acc

# Predictive system
input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)
# to numpy array
input_data = np.asarray(input_data)
# reshape
input_data = input_data.reshape(1,-1)

prediction = model.predict(input_data)
print(prediction)
if (prediction[0] == 0):
  print('The tumor is Benign')
elif (prediction[0] == 1):
  print('The tumor is Malignant')

#standardize
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_test_std = scaler.transform(X_test)

X_test_std

import keras
import tensorflow as tf

model2 = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model2.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')

loss, accuracy = model2.evaluate(X_test_std, Y_test)
print(accuracy)

Y_pred = model2.predict(X_test_std)

Y_pred

print(Y_pred.shape)
print(Y_pred[0])

Y_pred_diagnosis = [np.argmax(i) for i in Y_pred]
print(Y_pred_diagnosis)

input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model2.predict(input_data_std)

prediction_label = [np.argmax(prediction)]

if(prediction_label[0] == 1):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')

import joblib
joblib.dump(model,'L_Model.h5')
model2.save('NN_Model.h5')