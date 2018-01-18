from numpy import *
import operator
import os

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #二维数组，求数组的行数
    dataSetSize = dataSet.shape[0]
    #二维数组，求数组的列数
    #dataSetCol = dataSet.shape[1]
    print(dataSetSize)
    # numpy.tile([1, 2], 3)==>array([1, 2, 1, 2, 1, 2])
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #平方
    sqlDiffMat = diffMat ** 2
    #axis=1,行向量求和； axis=0， 列向量求和
    sqlDistances = sqlDiffMat.sum(axis = 1)
    #开方
    distances = sqlDistances ** 0.5
    #argsort 返回数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        print(voteIlabel)
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        print(classCount[voteIlabel])
        #print(classCount[voteIlabel])
    #operator.itemgetter(1)获取对象的第1个域的值
    #operator.itemgetter函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#python3把reload内置函数移到了imp标准库模块中
#from imp import reload
#reload(KNN)

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    print(returnMat)
    fr.close()
    fr = open(filename)
    classLabelVector = []
    index = 0
    for line in fr.readlines():
        # 移除头尾的空格
        line = line.strip()
        #print(line)
        listFromLine = line.split("\t")
        #数组中>=0 and <3的区间
        #print(listFromLine[0: 3])
        returnMat[index:] = listFromLine[0:3]
        # 数组中最后一个数
        #print(listFromLine[-1])
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # dataSet各个维度的长度，shape(array([1,1],[1,2],[1,3],[1,4]))=(4,2)
    #print(shape(dataSet))
    #numpy.zeros, zeros(5)=array([0.,0.,0.,0.,0.]), zeros(2,2)=array([[0.,0.],[0.,0.]])
    normDataSet = zeros(shape(dataSet))
    # shape[0] 表示第一维的长度，即行数
    m = dataSet.shape[0]
    #tile(A,B),把A重复B次，在行方向上重复m次，列方向上重复1次
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #print(normMat)
    print("ranges:", ranges) #ranges: [  9.12730000e+04   2.09193490e+01   1.69436100e+00]
    print("minVals:", minVals) #minVals: [ 0.        0.        0.001156]
    #行数
    m = normMat.shape[0]
    print("m:", m) # 1000
    numTestVecs = int(m*hoRatio)
    print("numTestVecs:", numTestVecs) # 100
    errorCount = 0.0
    for i in range(numTestVecs):
        #print(normMat[i, :])
        #print(normMat[numTestVecs:m, :])
        #print(datingLabels[numTestVecs:m])
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        #print(f"the classifier came back with: {classifierResult}, the real answer is: {datingLabels[i]}")
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: {}".format(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: ', resultList[classifierResult - 1])

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('_')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileNameStr}')

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f"the classifier came back with: {classifierResult}, the real answer is: {classNumStr} ")

        if classifierResult != classNumStr:
            errorCount += 1.0
    print(f"\nthe total number of errors is: {errorCount}")
    print("\nthe total error rate is: {}".format(errorCount/float(mTest)))

#shape()
#shape是numpy函数库中的方法，用于查看矩阵或者数组的维素
#>>>shape(array) 若矩阵有m行n列，则返回(m,n)
#>>>array.shape[0] 返回矩阵的行数m，参数为1的话返回列数n
#tile()
#tile是numpy函数库中的方法，用法如下:
#>>>tile(A,(m,n))  将数组A作为元素构造出m行n列的数组
#sum()
#sum()是numpy函数库中的方法
#>>>array.sum(axis=1)按行累加，axis=0为按列累加
#argsort()
#argsort()是numpy中的方法，得到矩阵中每个元素的排序序号
#>>>A=array.argsort()  A[0]表示排序后 排在第一个的那个数在原来数组中的下标
#dict.get(key,x)
#Python中字典的方法，get(key,x)从字典中获取key对应的value，字典中没有key的话返回0
