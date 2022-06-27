#importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#Loading Data

data = pd.read_csv("phishing.csv")
data.head()

#Listing the features of the dataset

data.columns

#Information about the dataset

data.info()

#nunique values in columns

data.nunique()

#droping index column

data = data.drop(['Index'],axis = 1)

#description of dataset

data.describe().T


#Splitting the dataset into dependant and independant features

X = data.drop(["class"],axis =1)
y = data["class"]

#Splitting the dataset into train and test sets: 80-20 split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

#Model Building & Training
#Creating holders to store the model performance results

ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#function to call for storing the results

def storeResults(model, a,b,c,d):
    ML_Model.append(model)
    accuracy.append(round(a, 3))
    f1_score.append(round(b, 3))
    recall.append(round(c, 3))
    precision.append(round(d, 3))

#Gradient Boosting Classifier Model

from sklearn.ensemble import GradientBoostingClassifier

# instantiate the model

gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model

gbc.fit(X_train,y_train)

#predicting the target value from the model for the samples
    
y_train_gbc = gbc.predict(X_train)
y_test_gbc = gbc.predict(X_test)

#computing the accuracy, f1_score, Recall, precision of the model performance

acc_train_gbc = metrics.accuracy_score(y_train,y_train_gbc)
acc_test_gbc = metrics.accuracy_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
print()
f1_score_train_gbc = metrics.f1_score(y_train,y_train_gbc)
f1_score_test_gbc = metrics.f1_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
print("Gradient Boosting Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
print()
recall_score_train_gbc = metrics.recall_score(y_train,y_train_gbc)
recall_score_test_gbc =  metrics.recall_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
print("Gradient Boosting Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
print()
precision_score_train_gbc = metrics.precision_score(y_train,y_train_gbc)
precision_score_test_gbc = metrics.precision_score(y_test,y_test_gbc)
print("Gradient Boosting Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
print("Gradient Boosting Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))

#computing the classification report of the model

print(metrics.classification_report(y_test, y_test_gbc))

training_accuracy = []
test_accuracy = []

#try learning_rate from 0.1 to 0.9,
    
depth = range(1,10)
for n in depth:
	forest_test =  GradientBoostingClassifier(learning_rate = n*0.1)
	forest_test.fit(X_train, y_train)
	# record training set accuracy
	training_accuracy.append(forest_test.score(X_train, y_train))
	# record generalization accuracy
	test_accuracy.append(forest_test.score(X_test, y_test))


#plotting the training & testing accuracy for n_estimators from 1 to 50
    
plt.figure(figsize=None)
plt.plot(depth,training_accuracy,label="training accuracy")
plt.plot(depth,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("learning_rate")
plt.legend()


training_accuracy = []
test_accuracy = []

# try learning_rate from 0.1 to 0.9
    
depth = range(1,10,1)
for n in depth:
	forest_test =  GradientBoostingClassifier(max_depth = n,learning_rate = 0.7)
	forest_test.fit(X_train, y_train)
	# record training set accuracy	
	training_accuracy.append(forest_test.score(X_train, y_train))
	# record generalization accuracy
	test_accuracy.append(forest_test.score(X_test, y_test))

#plotting the training & testing accuracy for n_estimators from 1 to 50

plt.figure(figsize=None)
plt.plot(depth,training_accuracy,label="training accuracy")
plt.plot(depth,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")  
plt.xlabel("max_depth")
plt.legend()
plt.show()

