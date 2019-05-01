from math import log,sqrt
import operator
import re
import pandas as pd
import numpy as np
import re
import pydotplus
import pprint
def ent(data):
    feat = {}
    for feature in data:
        curlabel = feature[-1]
        if curlabel not in feat:
            feat[curlabel] = 0
        feat[curlabel] += 1
    s = 0.0
    num = len(data)
    for it in feat:
        p = feat[it] * 1.0 / num
        s -= p * log(p,2)
    return s

def createDataSet():
    data = np.array(pd.read_csv("glass.data",sep=",",header = None))
    data = data[:].tolist()
    label = [str(f_index) for f_index in range(0, len(data[0]))]
    return data, label

def item_count(data):
    item_dict = {}
    label_list = [data[i][-1] for i in range(0,len(data))]
    for label in label_list:
        item_dict[label] = label_list.count(label)
    return item_dict

def calcShannonEntropy(data):
    Entropy=0.0
    label_dict=item_count(data)
    for label_key in label_dict:
        prob=float(label_dict[label_key])/(len(data))
        Entropy -= prob*np.math.log(prob,2)
    return Entropy

def majorityClass(data):
    items = item_count(data)
    max_label = max(items,key = items.get)
    return max_label

def splitDataSet(dataSet,i,value):
    subDataSet=[]
    for one in dataSet:
        if one[i]==value:
            reduceData=one[:i]
            reduceData.extend(one[i+1:])
            subDataSet.append(reduceData)
    return subDataSet

def splitContinuousDataSet(data,i,value,direction):
    subDataSet=[]
    for col in data:
        if direction==0:
            if col[i]>value:
                reduceData=col[:i]
                reduceData.extend(col[i+1:])
                subDataSet.append(reduceData)
        if direction==1:
            if col[i]<=value:
                reduceData=col[:i]
                reduceData.extend(col[i+1:])
                subDataSet.append(reduceData)
    return subDataSet

def chooseBestFeat(dataSet,labels):
    baseEntropy=calcShannonEntropy(dataSet)
    baseGainRatio=-1
    bestSplitDic={}
    for num_data in range(len(dataSet[0][:-1])):
        features=[dataSet[x][num_data] for x in range(0,len(dataSet))]
        if type(features[0]).__name__=='int' or type(features[0]).__name__=='float':
            features_sorted=sorted(features)
            splitList=[]
            for feature_num in range(len(features)-1):
                splitList.append((features_sorted[feature_num]+features_sorted[feature_num+1])/2.0)
            for num in range(len(splitList)):
                newEntropy=0.0
                gainRatio=0.0
                splitInfo=0.0
                value=splitList[num]
                subDataSet0=splitContinuousDataSet(dataSet,num_data,value,0)
                subDataSet1=splitContinuousDataSet(dataSet,num_data,value,1)
                prob0=float(len(subDataSet0))/len(dataSet)
                newEntropy-=prob0*calcShannonEntropy(subDataSet0)
                prob1=float(len(subDataSet1))/len(dataSet)
                newEntropy-=prob1*calcShannonEntropy(subDataSet1)
                splitInfo = splitInfo - prob0*log(prob0,2) - prob1*log(prob1,2)
                gainRatio=float(baseEntropy-newEntropy)/splitInfo
                if gainRatio>baseGainRatio:
                    baseGainRatio=gainRatio
                    bestSplit=num
                    bestFeat=num_data
            bestSplitDic[labels[num_data]]=splitList[bestSplit]
        else:
            uniqueFeatVals=set(features)
            GainRatio=0.0
            splitInfo=0.0
            newEntropy=0.0
            for value in uniqueFeatVals:
                subDataSet=splitDataSet(dataSet,num_data,value)
                prob=float(len(subDataSet))/len(dataSet)
                splitInfo-=prob*log(prob,2)
                if (splitInfo > 0):
                    newEntropy-=prob*calcShannonEntropy(subDataSet)
                    gainRatio=float(baseEntropy-newEntropy)/splitInfo
                    if gainRatio > baseGainRatio:
                        bestFeat = num_data
                        baseGainRatio = gainRatio
    if type(dataSet[0][bestFeat]).__name__=='float' or type(dataSet[0][bestFeat]).__name__=='int':
        bestFeatValue=bestSplitDic[labels[bestFeat]]
    if type(dataSet[0][bestFeat]).__name__=='str':
        bestFeatValue=labels[bestFeat]
    return bestFeat,bestFeatValue


def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if len(set(classList))==1:
        return classList[0]
    if len(dataSet[0])==1:
        return majorityClass(dataSet)
    Entropy = calcShannonEntropy(dataSet)
    bestFeat,bestFeatLabel=chooseBestFeat(dataSet,labels)
    myTree={labels[bestFeat]:{}}
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat+1:])
    if type(dataSet[0][bestFeat]).__name__=='str':
        featVals = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featVals)
        for value in uniqueVals:
            reduceDataSet=splitDataSet(dataSet,bestFeat,value)
            myTree[labels[bestFeat]][value]=createTree(reduceDataSet,subLabels)
    if type(dataSet[0][bestFeat]).__name__=='int' or type(dataSet[0][bestFeat]).__name__=='float':
        value=bestFeatLabel
        greaterDataSet=splitContinuousDataSet(dataSet,bestFeat,value,0)
        smallerDataSet=splitContinuousDataSet(dataSet,bestFeat,value,1)

        myTree[labels[bestFeat]]['>' + str(value)] = createTree(greaterDataSet, subLabels)

        myTree[labels[bestFeat]]['<=' + str(value)] = createTree(smallerDataSet, subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    newlist = list()
    for i in inputTree.keys():
        newlist.append(i)
    firstStr = newlist[0]    
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    secondlist = list()
    for i in secondDict.keys():
        secondlist.append(i)
    for key in secondlist:
        if (("<=" in key) and (testVec[featIndex] <= (float(re.findall(r"\d+\.?\d*",key)[0])))) or\
             ((">" in key) and (testVec[featIndex] > (float(re.findall(r"\d+\.?\d*",key)[0]))))     : 
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:classLabel=secondDict[key]
    return classLabel   

def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        if classify(myTree,labels,data_test[i])!=data_test[i][-1]:
            error+=1
    print ('myTree %d' %error)
    return float(error)

if __name__ == '__main__':
    dataSet,labels=createDataSet()
    a = createTree(dataSet,labels)
    print(a)
    #testing(a,dataSet, labels)
    '''
    from sklearn.tree import DecisionTreeClassifier
    dtree=DecisionTreeClassifier(criterion='entropy')
    from sklearn import tree
    m = dtree.fit([dataSet[x][0:-1] for x in range(0,len(dataSet))],[dataSet[x][-1] for x in range(0,len(dataSet))])
    tree.export_graphviz(dtree,out_file='tree1.dot') 
    #画图方法1-生成dot文件
    with open('tree1.dot', 'w') as f:
      dot_data = tree.export_graphviz(dtree, out_file=None)
      f.write(dot_data)

    #画图方法2-生成pdf文件
    dot_data = tree.export_graphviz(dtree, out_file=None,feature_names=dtree.feature_importances_,
                                  filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    ###保存图像到pdf文件
    graph.write_pdf("treetwo1.pdf")
    #print (tree)
    '''






