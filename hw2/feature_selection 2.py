import numpy as np
from mlxtend.data import loadlocal_mnist
from scipy.spatial.distance import pdist
import cv2
import collections
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def KNN_TestResult(test_data, training_data, training_label, K):
  result_list = []
  a = 0
  for x in test_data:
    a += 1
    result = KNN_classifier(x, training_data, training_label, K)
    result_list.append(result)
  return np.array(result_list)
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

data1 = np.append(training_data,test_data,axis =0)
label = np.append(training_label,test_label,axis =0)
data = BDS_algorithm(data1,10,training_label)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[:10000], label[:10000]) 
test_result = neigh.predict(data[10000:])
test_score = neigh.score(data[10000:],label[10000:])
index = index_list(data1,data)
#save_feature_file(data,index)
'''

data = load_testing_file()[0:150]
new_array = np.zeros(784)
data = data.astype(int)
new_array[data] = 256
file_name = "recon_" + str(150) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)
'''
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
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
'''