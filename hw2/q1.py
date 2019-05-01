from mlxtend.data import loadlocal_mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import matplotlib.patches as mpatches


############################## Question1 ###############################
# download dataset
class_a = pd.read_csv('Data/q1_classA.csv',header=None)
class_b = pd.read_csv('Data/q1_classB.csv', header=None)

####### 1.1 #######
class_a = np.array(class_a)
class_b = np.array(class_b)
# normalize
class_a =

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
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1, alpha=0.7)
    elif (style == 4):
        plt.contour(X0, X1, y_predict_matrix, colors='indianred', linewidths=1, alpha=0.7)
    elif (style == 5):
        plt.contour(X0, X1, y_predict_matrix, colors='peru', linewidths=1, alpha=0.7)
    elif (style == 6):
        plt.contour(X0, X1, y_predict_matrix, colors='green', linewidths=1, alpha=0.7)

# ax = fig1.add_subplot(221, aspect='equal')
# plot_decision_boundary(data_all, label_all, style=1, c=0.001)
# plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
# plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
#
# ax = fig1.add_subplot(222, aspect='equal')
# plot_decision_boundary(data_all, label_all, style=1, c=0.01)
# plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
# plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
#
# ax = fig1.add_subplot(223, aspect='equal')
# plot_decision_boundary(data_all, label_all, style=1, c=0.1)
# plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
# plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
#
# ax = fig1.add_subplot(224, aspect='equal')
# plot_decision_boundary(data_all, label_all, style=1, c=1)
# plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
# plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
# plt.show()

####### 1.3 #######
ax = fig1.add_subplot(111, aspect='equal')
plot_decision_boundary(data_all, label_all, style=3, c=0.001)
plot_decision_boundary(data_all, label_all, style=4, c=0.01)
plot_decision_boundary(data_all, label_all, style=5, c=0.1)
plot_decision_boundary(data_all, label_all, style=6, c=1)
plt.scatter(class_a[:,0], class_a[:,1],s = 4,alpha = 1, label = "classA")
plt.scatter(class_b[:,0], class_b[:,1],s = 4,alpha = 1, label = "classB")
ML_class1 = mpatches.Patch(color='firebrick', label='c=0.001')
ML_class2= mpatches.Patch(color='indianred', label='c=0.01')
ML_class3 = mpatches.Patch(color='peru', label='c=0.1')
ML_class4 = mpatches.Patch(color='green', label='c=1')
plt.legend(handles=[ML_class1,ML_class2,ML_class3,ML_class4])
plt.title("SVM with different C value")
plt.show()


