import logRegres
from numpy import *
import matplotlib.pyplot as plt


dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr, labelMat)
print(weights)

logRegres.plotBestFit(matrix.getA(weights))