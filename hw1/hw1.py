import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import cv2
import collections
from numpy import *

# 1
#a. generate 1000 sample and plot the figures
def plot_sample_data(cov,mean,figure_name):
	x,y = np.random.multivariate_normal(cov,mean,1000).T
	fig = plt.figure(figure_name)
	plt.scatter(x,y, c= "red", s= 0.35)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title(figure_name)
	plt.show()
	return x,y
#b. plot the contour for generated samples
def plot_contour(x,y,cov,figure_name):
	fig = plt.figure(figure_name)
	cov_array = np.array(cov)
	eigenvalue,eigenvector = np.linalg.eig(cov_array)
	angle = math.atan2(eigenvector[0][1],eigenvector[0][0])
	angle_degree = angle/math.pi * 180
	ax = fig.add_subplot(111, aspect='equal')
	e = Ellipse(xy = (0,0), width = math.sqrt( (eigenvalue[1])) * 2, 
	height = math.sqrt(eigenvalue[0])   * 2, angle = angle_degree,fc='none', ls='solid', ec='black', lw='3.')
	#ax.plot(x,y,'.')
	plt.scatter(x,y, c= "red", s= 0.35)
	ax.add_artist(e)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title(figure_name)
	plt.show()
#c calculate sample covariance matrix
def calculate_cov(x,y):
	x_mean = np.mean(x)
	y_mean = np.mean(y)
	cov_11 = np.dot((x - x_mean),(x - x_mean).T)/999
	cov_12 = np.dot((x - x_mean),(y - y_mean).T)/999
	cov_21 = np.dot((y - y_mean),(x - x_mean).T)/999
	cov_22 = np.dot((y - y_mean),(y - y_mean).T)/999
	cov_array = np.array([[cov_11,cov_12],[cov_21,cov_22]])
	return cov_array

#2
# fucntion to calculate multivariate normal from given information
def calculate_mvn():
	avg_1, avg_2, avg_3 = [3, 2], [5, 4], [2, 5]
	cov_1, cov_2, cov_3 = [[1, -1], [-1, 2]], [[1, -1], [-1, 7]], [[0.5, 0.5], [0.5, 3]]
	mvn_1 = multivariate_normal(avg_1, cov_1)
	mvn_2 = multivariate_normal(avg_2, cov_2)
	mvn_3 = multivariate_normal(avg_3, cov_3)
	return mvn_1,mvn_2,mvn_3

# a. ML classifier
def ML_Classifier(data_set):
	label = []
	mvn_1,mvn_2,mvn_3 = calculate_mvn()
	for data in data_set:
		density_list = [mvn_1.pdf(data), mvn_2.pdf(data), mvn_3.pdf(data)]
		max_index = density_list.index(max(density_list))
		label.append(max_index + 1)
	return np.array(label) #return the label of data 

# a. MAP classifier
def MAP_Classifier(data_set):
	label = []
	mvn_1,mvn_2,mvn_3 = calculate_mvn()
	for data in data_set:
		post_list = [mvn_1.pdf(data) * 0.2, mvn_2.pdf(data) * 0.3, mvn_3.pdf(data) * 0.5]
		max_index = post_list.index(max(post_list))
		label.append(max_index + 1)
	return np.array(label) #return the label of data 

	#a. fucntion to plot decision boundary if certaion classifier
def plot_decision_boundary(pred_func,plot_name):
    x_min, x_max = -15, 15
    y_min, y_max = -15, 15
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.title(plot_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot_name)
    plt.show()
plot_decision_boundary(lambda x: ML_Classifier(x),"ML Decision Boundary")

    #b. generate certain amount of data and reture the test result
def generate_samples(cov, mean, num):
	x = np.random.multivariate_normal(mean,cov,num)
	map_result = MAP_Classifier(x)
	ml_result = ML_Classifier(x) 
	return map_result,ml_result

	# calculate the confusion matrix and error rate from the given data
def calculate_confMatrix():
	cov1 = [[1,-1],[-1,2]]
	mean1 = [3,2]
	map_result1, ml_result1 = generate_samples(cov1,mean1, 600)
	true_result1 = [1] * 600
	cov2 = [[1, -1],[-1, 7]]
	mean2 = [5, 4]
	map_result2, ml_result2 = generate_samples(cov2,mean2, 900)
	true_result2 = [2] * 900
	cov3 = [[0.5, 0.5],[0.5, 3]]
	mean3 = [2, 5]
	map_result3, ml_result3 = generate_samples(cov3,mean3, 1500)
	true_result3 = [3] * 1500
	map_result = map_result1.tolist() + map_result2.tolist() + map_result3.tolist()
	ml_result = ml_result1.tolist() + ml_result2.tolist() + ml_result3.tolist()
	True_result = true_result1 + true_result2 + true_result3
	ml_conf = confusion_matrix(True_result,ml_result)
	map_conf = confusion_matrix(True_result,map_result)
	ml_error = (3000 - ml_conf[0][0] - ml_conf[1][1] - ml_conf[2][2])
	print (ml_conf)
	print (ml_error / 3000)
	map_error = (3000 - map_conf[0][0] - map_conf[1][1] - map_conf[2][2])
	print (map_conf)
	print (map_error / 3000)

def plot_contour_a():
	cov1 = [[1,-1],[-1,2]]
	mean1 = [3,2]
	x1 = np.random.multivariate_normal(mean1,cov1,600)
	cov2 = [[1, -1],[-1, 7]]
	mean2 = [5, 4]
	x2 = np.random.multivariate_normal(mean2,cov2,900)	
	cov3 = [[0.5, 0.5],[0.5, 3]]
	mean3 = [2, 5]
	x3 = np.random.multivariate_normal(mean3,cov3,1500)
	x, y = x1.T
	fig = plt.figure(" Data Contour")
	cov_array = np.array(cov1)
	eigenvalue,eigenvector = np.linalg.eig(cov_array)
	angle = math.atan2(eigenvector[0][1],eigenvector[0][0])
	angle_degree = angle/math.pi * 180
	ax = fig.add_subplot(111, aspect='equal')
	e = Ellipse(xy = (3,2), width = math.sqrt( (eigenvalue[0])) * 2, 
	height = math.sqrt(eigenvalue[1])   * 2, angle = -angle_degree,fc='none', ls='solid', ec='black', lw='3.')
	plt.scatter(x,y, c= "red", s= 0.35)
	ax.add_artist(e)
	x, y = x2.T
	cov_array = np.array(cov2)
	eigenvalue,eigenvector = np.linalg.eig(cov_array)
	angle = math.atan2(eigenvector[0][1],eigenvector[0][0])
	angle_degree = angle/math.pi * 180
	ax = fig.add_subplot(111, aspect='equal')
	e = Ellipse(xy = (5,4), width = math.sqrt( (eigenvalue[0])) * 2, 
	height = math.sqrt(eigenvalue[1])   * 2, angle = -angle_degree,fc='none', ls='solid', ec='black', lw='3.')
	#ax.plot(x,y,'.')
	plt.scatter(x,y, c= "green", s= 0.35)
	ax.add_artist(e)

	x, y = x3.T
	cov_array = np.array(cov3)
	eigenvalue,eigenvector = np.linalg.eig(cov_array)
	angle = math.atan2(eigenvector[0][1],eigenvector[0][0])
	angle_degree = angle/math.pi * 180
	ax = fig.add_subplot(111, aspect='equal')
	e = Ellipse(xy = (2,5), width = math.sqrt( (eigenvalue[0])) * 2, 
	height = math.sqrt(eigenvalue[1])   * 2, angle = -angle_degree,fc='none', ls='solid', ec='black', lw='3.')
	#ax.plot(x,y,'.')
	plt.scatter(x,y, c= "yellow", s= 0.35)
	ax.add_artist(e)

	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Data Contour")
	plt.show()
#3
# load data from mnist and reture data and label
def loaddata_training():
    X, y = loadlocal_mnist(
        images_path='/Users/tianmingyang/Desktop/train-images-idx3-ubyte', 
        labels_path='/Users/tianmingyang/Desktop/train-labels-idx1-ubyte')
    return np.float32(np.array(X)),np.array(y)
training_data, training_label = loaddata_training()
def loaddata_test():
    X, y = loadlocal_mnist(
        images_path='/Users/tianmingyang/Desktop/t10k-images-idx3-ubyte', 
        labels_path='/Users/tianmingyang/Desktop/t10k-labels-idx1-ubyte')
    return np.float32(np.array(X)), np.array(y)
test_data,test_label = loaddata_test()

#a, c PCA classifier return the data and reconstruction data
def PCA_Algotithm(data, d):
    mean = np.mean(data,axis=0)   #calculate of data average
    normal_data = data - mean
    cov_matrix = np.cov(normal_data.T)  # calculate the cov
    eig_value, eig_vector=  np.linalg.eig(cov_matrix)  
    feature_rank = np.argsort(-eig_value) 
    selected_feature = np.matrix(eig_vector.T[feature_rank[:d]]) 
    finalData = normal_data * selected_feature.T 
    finalData = finalData.real
    reconData = (finalData * selected_feature) + mean
    return finalData, reconData.real

#b Propose a suitable d using proportion of variance (POV) =95%.
def calculate_POV(data):
    mean = np.mean(data,axis=0)   #calculate of data average
    normal_data = data - mean
    cov_matrix = np.cov(normal_data.T)  # calculate the cov
    eig_value, eig_vector=  np.linalg.eig(cov_matrix)  
    feature_rank = np.argsort(-eig_value) 
    sum_eig =  sum(eig_value) * 0.95
    count = 0
    sum_count = 0
    for index in feature_rank:
    	if sum_count < sum_eig:
    		sum_count += eig_value[index]
    		count +=1
    return count

# c calculate the mean square error
def Calculate_MSE(data,d):
	rec_data = PCA_Algotithm(data,d)[1]
	return mean_squared_error(rec_data,data)

# c plot mse vs. d
def plot_MSE(data):
	mse_list = []
	d_list = []
	for d in range(1,785,10):
		mse = Calculate_MSE(data,d)
		mse_list.append(mse)
		d_list.append(d)
	fig = plt.figure("MSE vs. d")
	plt.xlabel("d")
	plt.ylabel("MSE")
	plt.title("MSE vs. d")
	plt.plot(d_list,mse_list,linewidth=2.0)
	plt.show()

#d Reconstruct a sample from the class of number ‘5’ and show it as a ‘png’ image for d= {1, 10, 50, 250, 784}
def Reconstruct_Five(data):
	for d in [1,10,50,250,784]:
		file_name = "recon_" + str(d) + ".png"
		rec_data = PCA_Algotithm(data,d)[1]
		img = rec_data[0].reshape(28,28)
		cv2.imwrite(file_name, img)

#e For the values of d= {1, 2, 3, 4, …, 784} plot eigenvalues (y-axis) versus d (x-axis).
def plot_eigenvalue(data):
    mean = np.mean(data,axis=0)   #calculate of data average
    normal_data = data - mean
    cov_matrix = np.cov(normal_data.T)  # calculate the cov
    eig_value, eig_vector=  np.linalg.eig(cov_matrix)  
    feature_rank = np.argsort(-eig_value) 
    plt.figure("eigenvalue vs. d")
    x = range(1,785)
    y_list = []
    for f in feature_rank:
        y_list.append(eig_value[f])
    plt.xlabel("d")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue vs. d")
    plt.plot(x,y_list)
    plt.show()
#4.
#a. KNN classifier 
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

#a return the test result label for certain k
def KNN_TestResult(test_data, training_data, training_label, K):
	result_list = []
	a = 0
	for x in test_data:
		a += 1
		result = KNN_classifier(x, training_data, training_label, K)
		result_list.append(result)
	return np.array(result_list)

#a calculate the accuracy
def KNN_accuracy(test_data,training_data,training_label,K):
	KNN_result = KNN_TestResult(test_data, training_data, training_label, K)
	True_result = loaddata_test()[1]
	score = accuracy_score(True_result, KNN_result)
	print (KNN_result)
	print (True_result)
	return score

#b use pca first and then use knn and return the test accuracy
def test_sample_PCA(test_data,training_data,training_label,d,k):
	new_training = PCA_Algotithm(training_data,d)[1]
	accuracy = KNN_accuracy(test_data,new_training,training_label,k)
	print (accuracy)


'''
mean = [0,0]
cov_a = [[1,0],[0,1]]
cov_b = [[1,0.9],[0.9,1]]
x1,y1 = plot_sample_data(mean,cov_a,"Sample a")
x2,y2 = plot_sample_data(mean,cov_b,"Sample b")
plot_contour(x1,y1,cov_a,"contour a")
plot_contour(x2,y2,cov_b,"contour b")
print (calculate_cov(x1,y1))
print (calculate_cov(x2,y2))
plot_decision_boundary(lambda x: ML_Classifier(x),"ML Decision Boundary")
plot_decision_boundary(lambda x: MAP_Classifier(x),"MAP Decision Boundary")

calculate_confMatrix()
print(calculate_POV(training_data))
plot_MSE(training_data)
Reconstruct_Five(training_data)
plot_eigenvalue(training_data)

print ("k vs reported accuracy")
for k in [1,3,5,11]:
	print (k)
	print (KNN_accuracy(test_data,training_data,training_label,k))
print ("done")

print ("k vs reported accuracy")
for k in [1,3,5,11]:
	for d in [5,50,100,500]:
		print ("k -- " +str(k)+ "   d -- " +str(d))
		test_sample_PCA(test_data,training_data,training_label,d,k)
'''
plot_contour_a()

