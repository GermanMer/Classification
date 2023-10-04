#Logistic Regression is a variation of Linear Regression, used when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables. Logistic regression fits a special s-shaped curve by taking the linear regression function and transforming the numeric estimate into a probability.

#You'll create a model for a telecommunication company, to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.

#Let's first import required libraries:
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

################################################About the dataset###########################################################

#We will use a telecommunications dataset for predicting customer churn.

churn_df = pd.read_csv(r'D:\Germán\Desktop\Python Files\ChurnData.csv')

##############################################Data pre-processing and selection###################################################

#Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the skitlearn algorithm:
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

#Let's define X, and y for our dataset:

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

y = np.asarray(churn_df['churn'])

#Also, we normalize the dataset:
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

###################################################Train/Test dataset######################################################

#We split our dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

####################################Modeling (Logistic Regression with Scikit-learn)#############################################

#Let's build our model using LogisticRegression from the Scikit-learn package. This function implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find extensive information about the pros and cons of these optimizers if you search it in the internet.
#The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem of machine learning models. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization. Now let's fit our model with train set:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#Now we can predict using our test set:
yhat = LR.predict(X_test)
yhat

#predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
yhat_prob = LR.predict_proba(X_test)
yhat_prob

##########################################################Evaluation##############################################################

#####jaccard index#####

#Let's try the jaccard index for accuracy evaluation. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)
#0.7058823529411765 #Resultado

#####confusion matrix#####

#Another way of looking at the accuracy of the classifier is to look at confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix

conf_matrix = confusion_matrix(y_test, yhat)
print('Matriz de confusión:')
print(conf_matrix)

#Classification report
print (classification_report(y_test, yhat))
#                precision    recall  f1-score   support #Resultado
#
#           0       0.73      0.96      0.83        25 #Resultado
#           1       0.86      0.40      0.55        15 #Resultado
#
#    accuracy                           0.75        40 #Resultado
#   macro avg       0.79      0.68      0.69        40 #Resultado
#weighted avg       0.78      0.75      0.72        40 #Resultado
#Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
#Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
#The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.
#Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.

#####log loss#####

#Now, let's try log loss for evaluation. In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
from sklearn.metrics import log_loss
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob))
#0.6017092478101185 #Resultado

######################################################Practice################################################################

#Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value?

LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
#LogLoss: : 0.61 #Resultado

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
