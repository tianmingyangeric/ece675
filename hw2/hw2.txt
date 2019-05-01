# -*- coding: utf-8 -*-

import pandas as pd
import math
import numpy as np
import operator
import urllib
import re
import copy
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
min_Error = 1000
from mlxtend.data import loadlocal_mnist
from scipy.spatial.distance import pdist
import cv2
import collections
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import matplotlib.image as mpimg

def noise_algorithm(L,data,noise):
    label_list = [d[-1] for d in data]
    label_list = np.unique(label_list)
    num_noise = int(len(data) * L )
    index_list = range(0,len(data))
    noise_index = random.sample(index_list,num_noise)
    normal_index = list(set(list(index_list)).difference(set(noise_index)))
    if noise == 'a':
        for noise in noise_index:
            index = random.sample(normal_index,1)
            data[noise] = data[index]
            diff_label = list(set(list(label_list)).difference(set([data[noise][-1]])))
            data[noise][-1] = random.sample(diff_label,1)[0]
    if noise == 'b':
        for noise in noise_index:
            diff_label = list(set(list(label_list)).difference(set([data[noise][-1]])))
            data[noise][-1] = random.sample(diff_label,1)[0]
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    dataSet = data[index]
    return dataSet

def load_data(file):
    data = np.array(pd.read_csv(file,sep=",",header = None))
    data = data[:].tolist()
    label = [str(f_index) for f_index in range(0, len(data[0]))]
    data = np.array(data)
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    dataSet = data[index]
    return dataSet,label

def get_test_data(dataSet, rounds):
    len_test = int(math.ceil(len(dataSet) / 10))
    try:
        test_data = dataSet[range(rounds*len_test, (rounds+1)*len_test)]
        rest_data = np.delete(dataSet,[range(rounds*len_test, (rounds+1)*len_test)],axis = 0)
    except IndexError:
        test_data = dataSet[range(rounds*len_test, len(dataSet))]
        rest_data = np.delete(dataSet,[range(rounds*len_test, len(dataSet))],axis = 0)
    return rest_data, test_data

def get_valid_data(dataSet):
    data_len = len(dataSet)
    len_valid = int(round(data_len * 0.2))
    index_list = range(0,data_len)
    valid_list = random.sample(index_list,len_valid)
    train_list = list(set(index_list).difference(set(valid_list)))
    train_data = dataSet[train_list].tolist()
    valid_data = dataSet[valid_list].tolist()
    return train_data, valid_data

def item_count(data):
    item_dict = {}
    label_list = [data[i][-1] for i in range(0,len(data))]
    for label in label_list:
        item_dict[label] = label_list.count(label)
    return item_dict

def cal_Entropy(data):
    Entropy=0.0
    label_dict=item_count(data)
    for label_key in label_dict:
        prob=float(label_dict[label_key])/(len(data))
        Entropy -= prob*np.math.log(prob,2)
    return Entropy

def get_maxlabel(data):
    items = item_count(data)
    max_label = max(items,key = items.get)
    return max_label

def splitDataSet(dataSet,n,value):
    subData=[]
    for data in dataSet:
        if data[n]==value:
            reducedData=data[:n]
            reducedData.extend(data[n+1:])
            subData.append(reducedData)
    return subData

def splitContinuous(data,x,value,direction):
    subData=[]
    for col in data:
        if (direction==0 and col[x]>value) or (direction==1 and col[x]<=value):
            reducedData=col[:x]
            reducedData.extend(col[x+1:])
            subData.append(reducedData)
    return subData

def Split_List(feature, dataSet):
    label_list = [data[len(data)-1] for data in dataSet]
    sortIndex = np.array(feature).argsort()
    Feature_Vec, sortlabel = [np.array(array)[sortIndex] for array in [feature,label_list]]
    flag = sortlabel[0]
    num = 0
    split_Value = []
    for label in sortlabel:
        if label != flag:
            feature1,feature2 = [Feature_Vec[num-1]for number in [num-1,num]]
            split_Value.append((feature1+feature2)/2)
        num += 1
        flag = label
    return split_Value


def chooseBestFeat(dataSet,labels):
    Split_Dic={}
    base_Entropy=cal_Entropy(dataSet)
    baseGainRatio = float(-'inf')
    best_Split = float(-'inf')
    for x in range(len(dataSet[0]) - 1):
        feat_Values = dataSet[:,x]
        if type(feat_Values[0]).__name__ =='float':
            splitList = Split_List(feat_Values, dataSet)
            for y in splitList:
                Entropy, gainRatio, splitInfo=[0.0,0.0,0.0]
                subData0, subData1 = [splitContinuous(dataSet,i,y,x) for x in [0,1]]
                prob0,prob1 = [float(len(y))/len(dataSet) for y in [subData0,subData1]]
                Entropy += (prob0*cal_Entropy(subData0) + prob1*cal_Entropy(subData1))
                if (prob0 != 0):
                    split_Info -= prob0*math.log(prob0,2)
                if (prob1 != 0):
                    split_Info -= prob1*math.log(prob1,2)
                if (split_Info == 0):
                    continue
                gain_Ratio=float(base_Entropy - Entropy)/split_Info
                if gain_Ratio > baseGainRatio:
                    baseGainRatio=gain_Ratio
                    best_Split, bestFeature = y, x
            Split_Dic[labels[x]] = best_Split
        else:
            uniqueValues = set(feat_Values)
            split_Info = 0.00001
            Entropy = 0
            for value in uniqueValues:
                subData=splitDataSet(dataSet,x,value)
                prob=float(len(subData))/len(dataSet)
                split_Info -= prob*math.log(prob,2)
                Entropy += prob * cal_Entropy(subData)
            gain_Ratio=float(baseEntropy-Entropy)/split_Info
            if gain_Ratio > baseGainRatio:
                bestFeature = x
                baseGainRatio = gain_Ratio

    dataType = type(dataSet[0][bestFeature]).__name__
    if dataType=='float' or dataType=='int':
        bestFeatValue=Split_Dic[labels[bestFeature]]
    else : 
        bestFeatValue=labels[bestFeature]
    return bestFeature,bestFeatValue

def majorityCount(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount, key=classCount.get)

def test_Majority(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    return float(error)

def buildTree(dataSet, labels, data_validation, purning): 
    Decision_Tree = {}
    classList = [data[-1] for data in dataSet]
    if len(set(classList)) == 1: 
        return classList[0]
    elif len(dataSet[0]) == 1 : 
        return get_maxlabel(dataSet)
    elif len(dataSet[0]) == 2: 
        feature = [dataSet[:,0:1]]
        if len(set(feature)) == 1:
            return get_maxlabel(dataSet)
    Entropy = cal_Entropy(dataSet)
    labels_c = copy.deepcopy(labels) 
    best_f,Feat_Value=chooseBestFeat(dataSet,labels)
    subLab = labels[:best_f] + labels[best_f+1:]
    if type(dataSet[0][best_f]).__name__=='float':
        leftData, rightData=[splitContinuous(dataSet,best_f,value,x) for x in [0,1]]
        feature_value = round(Feat_Value, 5)
        if len(leftData) != 0 and len(rightData) != 0:
            [Decision_Tree[labels[best_f]][mark + str(feature_value)] for mark in ['>', '<=']] = \
                buildTree(data, subLab, data_validation) for data in [leftData,rightData]
        elif len(rightData) != 0: 
            return get_maxlabel(leftData)
        else: return get_maxlabel(rightData)
    if type(dataSet[0][best_f]).__name__=='str':
        for data in set([data[best_f] for data in dataSet]): 
            reducedData=splitDataSet(dataSet,best_f,data)
            Decision_Tree[labels[best_f]][data]=buildTree(reducedData,subLab, data_validation)
    test_Result = testing(Decision_Tree, data_validation, labels_c)
    test_Result_M = test_Majority(majorityCount(classList), data_validation)
    if (test_Result_M < min_Error) and (test_Result_M < test_Result):
        min_Error = test_Result_M
        return majorityCount(classList)
    try:    
        return Decision_Tree
    except:
        return Decision_Tree{labels[best_f]:{}}

def classifier(Decision_Tree,featureLabel,testVector):
    first_Str = [*Decision_Tree][0]
    feat_Idx = featureLabel.index(first_Str)
    second_Dict = Decision_Tree[first_Str]
    for key in [*second_Dict]:
        if not((key.find("<=") == -1) and (testVector[feat_Idx] <= (float(re.findall(r"\d+\.?\d*",key)[0])))) or\
             (not((key.find(">") == -1)) and (testVector[feat_Idx] > (float(re.findall(r"\d+\.?\d*",key)[0])))):
            if type(second_Dict[key]).__name__=='dict':
                Label=classifier(second_Dict[key],featureLabel,testVector)
            else:Label=second_Dict[key]
        elif (('b' in key) or ('o' in key) or ('x' in key)):
            if type(second_Dict[key]).__name__ == 'dict' and testVector[feat_Idx] == key: 
                Label = classifier(second_Dict[key], featureLabel, testVector) 
            elif testVector[feat_Idx] == key:
                Label = second_Dict[key] 
    try:
        return Label
    except:
        return 'None'

def Accuracy_score(Decision_Tree,test_data,label):
    label_list,real_label = [],[]
    for i in range(0,len(test_data)):
        data = test_data[i]
        labels = classifier(Decision_Tree, label, data)
        label_list.append(labels)
        real_label.append(data[-1])
    score = accuracy_score(real_label,label_list) 
    return score

def plot_figure():
    x=['5%','10%','15%']
    y1=[0.6500568181818182,0.5971590909090906,0.5124431818181817]
    y2 = [0.6823295454545456,0.6123295454545452,0.5094318181818184]
    plt.figure()
    plt.plot(x,y1, color="r", linestyle="-", marker="^", linewidth=1, label="noise_a")
    plt.plot(x,y2, color="b", linestyle="-", marker="s", linewidth=1, label="noise_b")
    plt.legend(loc='upper left', bbox_to_anchor=(0.3, 0.95))
    plt.xlabel("Noise Percentage")
    plt.ylabel("Accuracy")
    plt.title("Glass Data")
    plt.show()
    plt.savefig("glass.jpg")

def plot_figure2():
    x=['5%','10%','15%']
    y1=[0.7100398936170214,0.6006981382978723,0.5124431818181817]
    y2 = [0.7564007092198578,0.6804011524822695,0.6192575354609928]
    plt.figure()
    plt.plot(x,y1, color="r", linestyle="-", marker="^", linewidth=1, label="noise_a")
    plt.plot(x,y2, color="b", linestyle="-", marker="s", linewidth=1, label="noise_b")
    plt.legend(loc='upper left', bbox_to_anchor=(0.3, 0.95))
    plt.xlabel("Noise Percentage")
    plt.ylabel("Accuracy")
    plt.title("Tic Data")
    plt.show()
    plt.savefig("glass.jpg")

for file_name in ["glass.data","tic-tac-toe.data"]:
    sumAccuracy = []
    for i in range(10):
        dataSet,label = load_data(file_name)
        dataSet = noise_algorithm(0.15,dataSet,'b')
        for j in range(10):
            rest, test = get_test_data(dataSet, j)
            train, valid = get_valid_data(rest)
            tree = createTree(train, label, valid)
            sumAccuracy.append(Accuracy_score(tree, test, label))
    variance = np.var(sumAccuracy)
    print (sum(sumAccuracy) / 100)
    print (variance)
    plot_figure2()
    plot_figure()

def loaddata_training():
X, y = loadlocal_mnist(
    images_path='/Users/tianmingyang/Desktop/train-images-idx3-ubyte', 
    labels_path='/Users/tianmingyang/Desktop/train-labels-idx1-ubyte')
return np.float32(np.array(X))[0:10000],np.array(y)[0:10000]

def loaddata_test():
    X, y = loadlocal_mnist(
        images_path='/Users/tianmingyang/Desktop/t10k-images-idx3-ubyte', 
        labels_path='/Users/tianmingyang/Desktop/t10k-labels-idx1-ubyte')
    return np.float32(np.array(X)), np.array(y)
training_data, training_label = loaddata_training()
test_data,test_label = loaddata_test()

def Euclidean_distace(vec1,vec2):
  distance = np.linalg.norm(vec1 - vec2)
  return distance

def interclass_distance(dataset, labels):
  feature_mean = []
  distance_list = []
  total_mean = []
  for label in set(labels):
    mean_list = []
    indices = [m for m, x in enumerate(labels) if x == label]
    data = dataset[indices]
    feature_list = data.T
    for feature in data.T:
      mean_list.append(np.mean(feature))
    mean_list = np.array(mean_list)
    feature_mean.append(mean_list)
  for f in np.array(feature_mean).T:
    total_mean.append(np.mean(f))
  total_mean = np.array(total_mean)
  for x in range(0,len(feature_mean)):
    distance_list.append(Euclidean_distace(feature_mean[x],total_mean))
  return (sum(distance_list))


def sfs_algorithm(dataset,labels,selected_list):
  grade_list = []
  feature_set = dataset.T
  for feature in dataset.T:
    selected_list = np.append(selected_list,[feature],axis =0)
    grade = interclass_distance(selected_list.T,labels)
    grade_list.append(grade)
    selected_list = np.delete(selected_list,-1,0)
  index = grade_list.index(max(grade_list))
  selected_list = np.append(selected_list,[dataset.T[index]],axis =0)
  feature_set = np.delete(feature_set,index,0)
  return selected_list,feature_set.T

def sbs_algorithm(dataset,selected_list,labels):
  grade_list = []
  feature_set = dataset.T
  for feature in dataset.T:
    selected_list = np.append(selected_list,[feature],axis =0)
    grade = interclass_distance(selected_list.T,labels)
    grade_list.append(grade)
    selected_list = np.delete(selected_list,-1,0)
  index = grade_list.index(min(grade_list))
  feature_set = np.delete(feature_set,index,0)
  return feature_set.T

def BDS_algorithm(dataset,d,labels):

  selected_list = np.empty(shape=[0,len(dataset)])
  feature_len = len(dataset.T)
  reduce_count = feature_len - d
  for count in range(0,d):
    selected_list,dataset = sfs_algorithm(dataset,labels,selected_list)
    if reduce_count > 0:
      dataset = sbs_algorithm(dataset,selected_list,labels)
      reduce_count -= 1
  return np.array(selected_list).T

def KNN_classifier(test_data, training_data, labels, k):
    distance_list = []
    targets = []
    for x_train in range(len(training_data)):
        distance = np.sqrt(np.sum(np.square(test_data - training_data[x_train, :])))
        distance_list.append([distance,x_train])
    distance_list = sorted(distance_list)
    for m in range(0,k):
        index = distance_list[m][1]
        targets.append(labels[index])
    return collections.Counter(targets).most_common(1)[0][0]

def KNN_accuracy(test_data,test_label,training_data,training_label,K):
  KNN_result = KNN_TestResult(test_data, training_data, training_label, K)
  True_result = test_label
  score = accuracy_score(True_result, KNN_result)
  return score

def plot_sample_data(cov,mean,figure_name):
    x,y = np.random.multivariate_normal(cov,mean,1000).T
    fig = plt.figure(figure_name)
    plt.scatter(x,y, c= "red", s= 0.35)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(figure_name)
    plt.show()
    return x,y

def calculate_cov(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_11 = np.dot((x - x_mean),(x - x_mean).T)/999
    cov_12 = np.dot((x - x_mean),(y - y_mean).T)/999
    cov_21 = np.dot((y - y_mean),(x - x_mean).T)/999
    cov_22 = np.dot((y - y_mean),(y - y_mean).T)/999
    cov_array = np.array([[cov_11,cov_12],[cov_21,cov_22]])
    return cov_array    

def Reconstruct_Five(data):
    for d in [1,10,50,250,784]:
        file_name = "recon_" + str(d) + ".png"
        rec_data = PCA_Algotithm(data,d)[1]
        img = rec_data[0].reshape(28,28)
        cv2.imwrite(file_name, img)    
        
def plot_data(cov,figure_name):
    x,y = np.random.multivariate_normal(cov,1,1000).T
    fig = plt.figure(figure_name)
    plt.scatter(x,y, c= "red", s= 0.35)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(figure_name)
    plt.show()
    return x,y

def KNN_TestResult(test_data, training_data, training_label, K):
  result_list = []
  a = 0
  for x in test_data:
    a += 1
    result = KNN_classifier(x, training_data, training_label, K)
    result_list.append(result)
  return np.array(result_list)

def KNN_accuracy(test_data,training_data,training_label,K):
    KNN_result = KNN_TestResult(test_data, training_data, training_label, K)
    True_result = loaddata_test()[1]
    score = accuracy_score(True_result, KNN_result)
    print (KNN_result)
    print (True_result)
    return score

def index_list(origin_data, reduced_data):
  index_list = []
  for data in reduced_data.T:
    index = origin_data.T.tolist().index(data.tolist())
    index_list.append(index)
  return index_list
def save_feature_file(data,index):

  np.save("training_data_384",data)
  np.save("index_data",index)

def load_testing_file():
  feature = np.load("index_data.npy")
  return feature

def save_feature_file_2():
    """ 
    save generated feature array to csv file
    """
    training_features,training_label = generate_features("./training_data/")
    test_features,test_label = generate_features("./test_data/")
    np.save("training_data",training_features)
    np.save("training_label",training_label)
    np.save("test_data",test_features)
    np.save("test_label",test_label)

def Reconstruct_Five(data):
    for d in [1,10,50,250,784]:
        file_name = "recon_" + str(d) + ".png"
        rec_data = PCA_Algotithm(data,d)[1]
        img = rec_data[0].reshape(28,28)
        cv2.imwrite(file_name, img)

def load_training_file():
    """ 
    read generated feature array from csv file
    """
    feature = np.load("training_data.npy")
    label = np.load("training_label.npy")
    return feature,label

data1 = np.append(training_data,test_data,axis =0)
label = np.append(training_label,test_label,axis =0)
data = BDS_algorithm(data1,10,training_label)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[:10000], label[:10000]) 
test_result = neigh.predict(data[10000:])
test_score = neigh.score(data[10000:],label[10000:])
index = index_list(data1,data)
save_feature_file(data,index)
data = load_testing_file()[0:150]
new_array = np.zeros(784)
data = data.astype(int)
new_array[data] = 256
file_name = "recon_" + str(150) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)

def LDA_algorithm(data, labels, k):
    label_list = list(set(labels))
    X_classifier = {}
    mean_classifier = {}

    for label in set(labels):
      indices = [m for m, x in enumerate(labels) if x == label]
      X_classifier[label] = np.array(data[indices])
      mean_classifier[label] = np.mean(X_classifier[label], axis=0)
    mean_list =  [np.mean(feature) for feature in data.T]
    Sb, Sw = np.zeros((784, 784)), np.zeros((784, 784))
    for i in label_list:
        Sw += np.dot((X_classifier[i] - mean_classifier[i]).T,
                     X_classifier[i] - mean_classifier[i])
        Sb += len(X_classifier[i]) * np.dot((mean_classifier[i] - mean_list).reshape(
            (len(mean_list), 1)), (mean_classifier[i] - mean_list).reshape((1, 784)))
    eig_value, eig_vector = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    sorted_indices = np.argsort(eig_value)
    topk_eig_vecs = eig_vector[:, sorted_indices[:-k - 1:-1]]
    return topk_eig_vecs.real
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def test_sample_LDA(test_data,training_data,training_label,d,k):
    new_training = LDA_Algotithm(training_data,d)[1]
    accuracy = KNN_accuracy(test_data,new_training,training_label,k)
    print (accuracy)
    
# LDA
sklearn_lda = LDA(n_components=5)
w = sklearn_lda.fit_transform(data1, label)


w = LDA_algorithm(data1,label,10)
train_data = np.dot(training_data,w)
test_data = np.dot(test_data,w)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data, training_label) 
test_result = neigh.predict(test_data)
test_score = neigh.score(test_data,test_label)
print(test_score,'test_score')






