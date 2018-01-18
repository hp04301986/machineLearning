from math import log
import numpy as np
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    #print("numEntries: ", numEntries)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        #print("currentLabel: ", currentLabel)
        #print("labelCounts.keys(): ", labelCounts.keys())
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        #print("labelCounts[key]: ", labelCounts[key])
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#在dataset中，根据第axis+1列的value划分dataset
def splitDataSet(dataSet, axis, value):
    refDataSet = []
    for featVec in dataSet:
        #print("featVec: ", featVec)
        #print("axis: ", axis)
        #print("value: ", value)
        if featVec[axis] == value:
            #print(array[:-1]) 输出-1之前的元素，即最后一个元素之前的
            reducedFeatVec = featVec[: axis]
            #print("reducedFeatVec: ",reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1 :])
            #print("reducedFeatVecExtend: ", reducedFeatVec)
            refDataSet.append(reducedFeatVec)
            #print("refDataSet: ", refDataSet)
    print("refDataSet:  ", refDataSet)
    return refDataSet
#????
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    print("baseEntropy:", baseEntropy)
    bestInfoGain = 0.0
    bestFeature = -1
    tempSet = dataSet
    mtx = np.matrix(tempSet)
    #print(mtx)
    for i in range(numFeatures):
        #获取第i个特征所有的可能取值
        featList = [example[i] for example in dataSet]
        #另一种写法：featList = mtx.T[i].tolist()[0]
        #print(featList)
        uniqueVals = set(featList)
        #print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            #print(value)
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            #print(prob)
            #计算新熵值
            newEntropy += prob * calcShannonEnt(subDataSet)
            #print(newEntropy)
        infoGain = baseEntropy - newEntropy
        if (infoGain >= bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    print(bestFeature)
    return bestFeature

#该函数使用分类名称的列表，然后创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个类标签出现的频率，返回次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    #print("classList: ", classList)
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #print("classCount: ", classCount)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    print("labels: ", labels)
    #最后一个特征的可能取值
    classList = [example[-1] for example in dataSet]
    print("classList: ", classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print("bestFeat: ", bestFeat)
    bestFeatLabel = labels[bestFeat]
    print("bestFeatLabel: ", bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    print("myTree: ", myTree)
    del(labels[bestFeat])
    print("labels again: ", labels)
    featValues = [example[bestFeat] for example in dataSet]
    print("featValues: ", featValues)
    #[1, 0]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        print("subLabels: ", subLabels)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    print("testVec: ", testVec)
    firstStr = list(inputTree.keys())[0] # no surfacing
    print("firstStr: ", firstStr)
    secondDict = inputTree[firstStr]
    print("secondDict: ", secondDict)
    print("featLabels: ", featLabels)
    featIndex = featLabels.index(firstStr) # 0
    print("featIndex: ", featIndex)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                print("key: ", key)
                print("secondDict[key]: ", secondDict[key])
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                print("key: ", key)
                classLabel = secondDict[key]
    print("classLabel: ", classLabel)
    return classLabel
