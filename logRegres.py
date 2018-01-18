from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

#返回训练好的回归系数
def gradAscent(dataMatIn, classLabels):
    #print("dataMatIn: ", dataMatIn)
    dataMatrix = mat(dataMatIn)
    #print("dataMatrix: ", dataMatrix)
    labelMat = mat(classLabels).transpose()#矩阵转置
    print("labelMat: ", labelMat)
    m, n = shape(dataMatrix)
    print("m: ", m)
    print(f"n: {n}")
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    print(f"weights: {weights}")
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        print("h: ", h)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    print("final weights: ", weights)
    return weights

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    print("n: ", n)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #print("x: ", x)
    y = (- weights[0] - weights[1] * x) / weights[2]
    #print("y: ", y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法 dataMatrix是数组
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    print(weights)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        #print("h====", h)
        #print("classLabels[i]====", classLabels[i])
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 1000):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m)) #python3.x is different from python 2.x dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            #删除之后再进行下一次迭代
            del(dataIndex[randIndex]) #python3 里面不支持range()里面的删除
    return weights
