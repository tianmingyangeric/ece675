from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import random

def load_data_a():

    dataA=pd.read_csv('q1_classA.csv', sep=',',header=None)
    dataB=pd.read_csv('q1_classB.csv', sep=',',header=None)
    return dataA,dataB

def load_data():
	dataA=pd.read_csv('classA.csv', sep=',',header=None)
	dataB=pd.read_csv('classB.csv', sep=',',header=None)
	return dataA,dataB

def get_data(dataA,dataB):
	labelA = np.repeat(0, len(dataA))
	labelB = np.repeat(1, len(dataB))
	total_data = np.vstack((dataA,dataB))
	label = np.append(labelA, labelB, 0)
	new_data = np.vstack((total_data.T,label.T)).T
	new_data = new_data.astype(np.float64)
	return new_data

def ploat_data(dataA,dataB):
    x1=dataA[0]
    y1=dataA[1]
    x2=dataB[0]
    y2=dataB[1] 
    plt.figure()
    plt.scatter(x1,y1, color="r", linestyle="-", marker="^", label="classA")
    plt.scatter(x2,y2, color="b", linestyle="-", marker="s", label="classB")
    plt.legend(loc='left', bbox_to_anchor=(0.3, 0.95))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("DATA")
    plt.show()

def SVM_accuracy(c_value,dataA,dataB):
	new_data =get_data(dataA,dataB)
	np.random.shuffle(new_data)
	clf = SVC(kernel='linear', C=c_value)
	data = [x[0:-1] for x in new_data]
	label = [x[-1] for x in new_data]
	scores = cross_val_score(clf, data, label, cv=10, scoring='accuracy')
	return (np.mean(scores))

def SVM_algorithm(c_value,train,label,test):
	clf = SVC(gamma='auto',C = c_value)
	clf.fit(train, label)
	return clf.predict(test)


def plot_decision_boundary(plot_name,c_value,dataA,dataB):
    x1=dataA[0]
    y1=dataA[1]
    x2=dataB[0]
    y2=dataB[1] 
    x_min, x_max = 150, 500
    y_min, y_max = 0, 350
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    fig1 = plt.figure(0)
    c_value = [0.001,0.01,0.1,1]
    location = [221,222,223,224]
    title = ['c = 0.001','c=0.01','c=0.1','c=1']
    for x in range(0,4):
        fig1.add_subplot(location[x], aspect='equal')
        clf = SVC(kernel='linear', C = c_value[x])
        print (c_value[x])
        new_data =get_data(dataA,dataB)
        data = [x[0:-1] for x in new_data]
        label = [x[-1] for x in new_data]
        clf.fit(data,label)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.title(plot_name)
        plt.scatter(x1,y1, color="r", linestyle="-", marker="^", label="classA")
        plt.scatter(x2,y2, color="b", linestyle="-", marker="s", label="classB")
        plt.title(title[x])
    plt.show()
#question 1
#a.
dataA, dataB = load_data_a()
ploat_data(dataA,dataB)
#b.
new_data =get_data(dataA, dataB)
data = [x[0:-1] for x in new_data]
data = np.array(data)
label = [x[-1] for x in new_data]
label = np.array(label)
dataA, dataB = load_data()
plot_decision_boundary("0.001",0.001,dataA,dataB)



#question 2
#a

dataA, dataB = load_data()
ploat_data(dataA,dataB)

#b

new_data =get_data(dataA, dataB)
data = [x[0:-1] for x in new_data]
data = np.array(data)
label = [x[-1] for x in new_data]
label = np.array(label)

for c in [0.1,1,10,100]:
    print (c)
    score = []
    for x in range(0,10):
        score.append(SVM_accuracy(c,dataA,dataB))
    print (np.mean(score))


#print("test accuracy is ",adaboost.score(X_test,Y_test))
#plot_decision_boundary("SVM Decision Boundary",1,dataA,dataB)
#ploat_data(dataA,dataB)
#SVM_accuracy(1,dataA,dataB)
#print (predict(X_train,Y_train,clf, alpha_arr,clf_arr))

def adaboost(X_train, Y_train, X_test, clf):
    len_train, len_test = len(X_train), len(X_test)
    weight = np.repeat(1/len_train,len_train)
    total_test = np.zeros(len_test)
    error = 0.5
    for i in range(0,50):
        value = random.sample(range(len(X_train)), 100)
        new_Y = Y_train[value].reshape(100,1)
        new_train = X_train[value]
        wei = weight[value]
        clf.fit(new_train, new_Y, sample_weight=wei)
        pred_train_new,pred_test_new = clf.predict(new_train),clf.predict(X_test)
        list_a = []
        for x in range(len(pred_train_new)):
            if pred_train_new[x] == new_Y[x]:
                list_a.append(0)
            else:
                list_a.append(1)
        list_a = np.array(list_a)
        A_B = []
        for d in pred_test_new:
            if d == 1:
                A_B.append(1)
            else:
                A_B.append(-1)
        A_B = np.array(A_B)
        err_amount = np.dot(wei, list_a)
        if err_amount <= error:
            beta = err_amount/(1 - err_amount) 
            wei = np.multiply(wei, [np.power(beta,(1-x)) for x in list_a])/sum(wei) 
            weight[value] = wei / sum(wei) 
            total_test += [x * np.math.log(1 / beta) for x in A_B]
    pred_test = np.where(np.sign(total_test) > 0, np.sign(total_test), 0)
    return np.array(pred_test)

clf = SVC(kernel='linear', C=1)
X_train, X_test, y_train, y_test = train_test_split(data,label,test_size = 0.1)
result = adaboost(X_train,y_train,X_test,clf)
print (accuracy_score(result,y_test))

def decision_boundary(plot_name,dataA,dataB):
    x1=dataA[0]
    y1=dataA[1]
    x2=dataB[0]
    y2=dataB[1] 
    x_min, x_max = 150, 500
    y_min, y_max = 0, 350
    h = 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    fig1 = plt.figure(0)
    new_data =get_data(dataA,dataB)
    data = [x[0:-1] for x in new_data]
    data = np.array(data)
    label = [x[-1] for x in new_data]
    label = np.array(label)
    clf = SVC(kernel='linear', C = 1)
    Z = adaboost(data,label,np.c_[xx.ravel(), yy.ravel()],clf)
    #Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.title(plot_name)
    plt.scatter(x1,y1, color="b", linestyle="-", marker="^", label="classA")
    plt.scatter(x2,y2, color="r", linestyle="-", marker="s", label="classB")
    plt.title("Decision Boundary")
    plt.show()
decision_boundary("SVM Decision Boundary",dataA,dataB)
