
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix

def load_data():
    dataA=pd.read_csv('classA.csv', sep=',',header=None)
    dataB=pd.read_csv('classB.csv', sep=',',header=None)
    return dataA,dataB

def get_data(dataA,dataB):
    labelA = np.repeat(1, len(dataA))
    labelB = np.repeat(2, len(dataB))
    total_data = np.vstack((dataA,dataB))
    label = np.append(labelA, labelB, 0)
    new_data = np.vstack((total_data.T,label.T)).T
    new_data = new_data.astype(np.float64)
    return new_data

dataA, dataB = load_data()
new_data =get_data(dataA, dataB)
data = [x[0:-1] for x in new_data]
data = np.array(data)
label = [x[-1] for x in new_data]
label = np.array(label)


X_train,X_test,Y_train,Y_test = train_test_split(data,label,test_size = 0.2,random_state = 1)


plot_decision_boundary("SVM Decision Boundary",1,dataA,dataB)
#ploat_data(dataA,dataB)
#SVM_accuracy(1,dataA,dataB)

def fit(data,label,c_value):
    clf = SVC(kernel='linear', C = c_value)
    alpha_arr = []
    clf_arr = []
    data_len = len(data)
    weight = np.repeat(1/data_len,data_len)
    for i in range(5):
        clf.fit(data, label, sample_weight=weight)
        clf_arr.extend([clf])
        Y_pred = clf.predict(data)
        indic_arr = [1 if Y_pred[i] != label[i] else 0 for i in range(data_len)]
        err = np.dot(weight, np.array(indic_arr))
        alpha = (1-err)/err
        alpha_arr.extend([alpha])
        temp = weight * (alpha**[1-i for i in indic_arr])
        weight = temp / np.sum(temp)
    return clf, alpha_arr,clf_arr

clf, alpha_arr,clf_arr = fit(data,label,1)

def predict(data,label,clf, alpha_arr,clf_arr):
    data_len = len(data)
    mulit_idx_pred = []
    mulit_Y_pred = []
    for i in range(5):
        Y_pred = clf_arr[i].predict(data)
        mulit_Y_pred.append(Y_pred.tolist())
        indic_arr = [1 if Y_pred[i] == label[i] else 0 for i in range(data_len)]
        temp = [np.log(alpha_arr[i])*k for k in indic_arr]
        mulit_idx_pred.append(temp)
    max_idx_pred = np.array(mulit_idx_pred).argmax(axis=0)
    mulit_Y_pred = np.array(mulit_Y_pred).T
    result = np.array([x[max_idx_pred[i]] for i,x in enumerate(mulit_Y_pred)],dtype=np.float64)
    score = accuracy_score(label, result)
    return score
#print (predict(X_train,Y_train,clf, alpha_arr,clf_arr))
