import logRegres
from numpy import *
import matplotlib.pyplot as plt


dataArr, labelMat = logRegres.loadDataSet()
# 梯度上升
#weights = logRegres.gradAscent(dataArr, labelMat)
#随机梯度上升
#weights = logRegres.stocGradAscent0(array(dataArr), labelMat)
#改进的随机梯度上升算法
#weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
#print(weights)

#logRegres.plotBestFit(matrix.getA(weights))
logRegres.multiTest()