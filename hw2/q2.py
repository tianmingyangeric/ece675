from mlxtend.data import loadlocal_mnist
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score




############################## Question2 ###############################
# download dataset
class_a = pd.read_csv('Data/classA.csv',header=None)
class_b = pd.read_csv('Data/classB.csv', header=None)

####### 1.1 #######
class_a = np.array(class_a)
class_b = np.array(class_b)

fig1 = plt.figure(0)
# ax = fig1.add_subplot(111, aspect='equal')
# plt.title('visualizing data set')
# plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
# plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
# ax.grid(True)
# plt.legend()
# plt.show()

####### 1.2 #######
label_a = np.zeros((len(class_a),1))
label_b = np.ones((len(class_b),1))
data_all = np.vstack((class_a,class_b))
label_all = np.vstack((label_a,label_b))

# shuffle data and label at the same time with same order
def ShuffleDataSet(data, label):
    all_data = np.hstack((data, label))
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    shuffled_all_data = all_data[indices]
    shuffled_label = shuffled_all_data[:,-1]
    shuffled_label = shuffled_label.reshape(len(shuffled_label),-1) # transpose
    shuffled_data = data[indices]
    return shuffled_data, shuffled_label

def SplitTestTrain(data, label, it):
    dataSet = np.hstack((data, label))
    len_data = len(dataSet)
    len_test = int(math.floor(len_data * 0.1))
    start = it * len_test
    finish = (it+1) * len_test
    test_data_all = dataSet[start: finish]
    test_data = test_data_all[:, [0,1]]
    test_label = test_data_all[:, -1]
    rest1 = dataSet[:start]
    rest2 = dataSet[finish:]
    if len(rest1)==0:
        train_data_all = rest2
    else:
        train_data_all = np.vstack((rest1,rest2))
    train_data = train_data_all[:, [0, 1]]
    train_label = train_data_all[:, -1]
    return train_data, train_label, test_data, test_label


def plot_decision_boundary(data, label, style, c):
    # meshgrid function can draw grid on figure and return coordinate matrix
    x_min, x_max = min(data[:,0]) - 10, max(data[:,0]) + 10
    y_min, y_max = min(data[:,1]) - 10, max(data[:,1]) + 10
    X0, X1 = np.meshgrid(
    # Randomly generate two sets of numbers, the starting value and density are determined by the starting value of the axis
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 0.5)
    )
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]

    # train and predict
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(data, label)
    y_predict = clf.predict(X_grid_matrix)
    y_predict_matrix = y_predict.reshape(X0.shape)

    # set the filled color
    from matplotlib.colors import ListedColormap
    my_colormap1 = ListedColormap(['lightcoral', 'mistyrose', 'pink'])
    my_colormap2 = ListedColormap(['deepskyblue', 'dodgerblue', 'skyblue'])

    # draw the Ml and MAP decision boundary with different color
    if (style == 2):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap2, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)
    elif (style == 1):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap1, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 3):
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 4):
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)

fig1 = plt.figure(0)
ax = fig1.add_subplot(221, aspect='equal')
plot_decision_boundary(data_all, label_all, style=1, c=0.1)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
ax.set_title('C = 0.1')

ax = fig1.add_subplot(222, aspect='equal')
plot_decision_boundary(data_all, label_all, style=1, c=1)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
ax.set_title('C = 1')

ax = fig1.add_subplot(223, aspect='equal')
plot_decision_boundary(data_all, label_all, style=1, c=10)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
ax.set_title('C = 10')

ax = fig1.add_subplot(224, aspect='equal')
plot_decision_boundary(data_all, label_all, style=1, c=100)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
ax.set_title('C = 100')
plt.show()




def SVMAccuracy(c,data,label):
    shuff_data, shuff_label = ShuffleDataSet(data,label)
    clf = svm.SVC(kernel='linear', C=c)
    accuracy = cross_val_score(clf, shuff_data, shuff_label, cv=10, scoring='accuracy')
    mean = np.mean(accuracy)
    # print (mean)
    return mean

accuracy_list1 = []
accuracy_list2 = []
accuracy_list3 = []
accuracy_list4 = []
for i in range (10):
    accuracy_list1.append(SVMAccuracy(0.1, data_all, label_all))
    # print(accuracy_list1)
    accuracy_list2.append(SVMAccuracy(1, data_all, label_all))
    # print(accuracy_list2)
    accuracy_list3.append(SVMAccuracy(10, data_all, label_all))
    # print(accuracy_list3)
    accuracy_list4.append(SVMAccuracy(100, data_all, label_all))
    # print(accuracy_list4)


print(np.mean(accuracy_list1))
print(np.mean(accuracy_list2))
print(np.mean(accuracy_list3))
print(np.mean(accuracy_list4))



# clf = svm.SVC(kernel='linear', C=0.1)
# adaboost = AdaBoost(50, clf)
# X_train, X_test, y_train, y_test = train_test_split(data_all, label_all, test_size=0.1, random_state=0)
# adaboost.fit(X_train, y_train)
#
# print("train accuracy is ",adaboost.score(X_train, y_train))
# print("test accuracy is ",adaboost.score(X_test, y_test))
#
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_train)
# print(accuracy_score(y_train, y_pred))
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

############################## Question3,4,5 ###############################
def plot_decision_boundary(data, label, style):
    # meshgrid function can draw grid on figure and return coordinate matrix
    x_min, x_max = min(data[:,0]) - 10, max(data[:,0]) + 10
    y_min, y_max = min(data[:,1]) - 10, max(data[:,1]) + 10
    X0, X1 = np.meshgrid(
    # Randomly generate two sets of numbers, the starting value and density are determined by the starting value of the axis
        np.arange(x_min, x_max, 1),
        np.arange(y_min, y_max, 0.5)
    )
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]

    # train and predict
    # clf = svm.LinearSVC(C=0.1)
    clf = svm.SVC(kernel='linear', C=1)
    y_predict = adaboost_clf(data, label, X_grid_matrix, 50, clf)
    y_predict_matrix = y_predict.reshape(X0.shape)

    # set the filled color
    from matplotlib.colors import ListedColormap
    my_colormap1 = ListedColormap(['lightcoral', 'mistyrose', 'pink'])
    my_colormap2 = ListedColormap(['deepskyblue', 'dodgerblue', 'skyblue'])

    # draw the Ml and MAP decision boundary with different color
    if (style == 2):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap2, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)
    elif (style == 1):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap1, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 3):
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 4):
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)

ax = fig1.add_subplot(221, aspect='equal')
plot_decision_boundary(X_train,y_train,1)
plt.show()


def adaboost_clf(X_train, Y_train, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w_list = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    test_sum = np.zeros(n_test)
    train_sum = np.zeros(n_train)
    for i in range(M):
        index = random.sample(range(len(X_train)), 100)
        X_train_m = X_train[index]
        Y_train_m = Y_train[index].reshape(100,1)
        w = w_list[index]
        clf.fit(X_train_m, Y_train_m, sample_weight=w)
        pred_train_i = clf.predict(X_train_m)
        pred_test_i = clf.predict(X_test)
        num = len(pred_train_i)
        miss = [1 if pred_train_i[x] != Y_train_m[x] else 0 for x in range(num)]
        A_or_B = [1 if item == 1 else -1 for item in pred_test_i]
        err_m = np.dot(w, miss)
        if err_m > 0.5:
            continue;
        beta_m = float(err_m)/float(1 - err_m) # Beta
        w = np.multiply(w, [beta_m**(1-x) for x in miss]) # if be classified wrong, the weight will not change
        normalized = sum(w)
        w = w / normalized # New weights
        w_list[index] = w

        #record the sum h for each class
        # Add to prediction
        test_sum_h_i = [x * math.log(1 / beta_m) for x in A_or_B]
        # test_sum_h_i = [x * math.log(1/beta_m) for x in pred_test_i]
        test_sum += test_sum_h_i
        # cause only have two classes, so we can use -1/1 to get the class label

    pred_test = np.sign(test_sum)
    pred_test = np.where(pred_test > 0, pred_test, 0)
    return np.array(pred_test)


############################## Question4 ###############################
def accuracy(pred, true):
    correct_num = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            correct_num += 1
    accur = correct_num/float(len(pred))
    print ("accuracy: " + str(accur))
    return accur


accuracy_list1 = []
clf = svm.SVC(kernel='linear', C=0.1)
for i in range (10):
    shuff_data, shuff_label = ShuffleDataSet(data_all, label_all)
    for j in range(10):
        X_train, y_train, X_test, y_test = SplitTestTrain(shuff_data, shuff_label, j)
        pred = adaboost_clf(X_train, y_train, X_test, 50, clf)
        accuracy_list1.append(accuracy(pred, y_test))
        # print(accuracy_list1)

print(np.var(accuracy_list1))
print(np.mean(accuracy_list1))

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


############################## Question5 ###############################

ax = fig1.add_subplot(111, aspect='equal')
plot_decision_boundary(data_all,label_all,1)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
plt.show()