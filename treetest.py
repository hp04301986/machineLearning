import trees
import treePlotter

myDat, labels = trees.createDataSet()
print(myDat)
#print(labels)

#print(trees.calcShannonEnt(myDat))

#myDat[0][-1] = 'maybe'
#print(myDat)
#print(trees.calcShannonEnt(myDat))

#trees.splitDataSet(myDat, 0, 1)
#trees.splitDataSet(myDat, 0, 0)
#trees.splitDataSet(myDat, 0, 0)
#myDat = [[1, 'yes'], [1, 'yes'], [1, 'no'], [0, 'no'], [0, 'no']]
#myDat = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 0, 'no'], [1, 0, 'no']]
#print(trees.chooseBestFeatureToSplit(myDat))
#print(myDat)
#print(trees.createTree(myDat, labels))
#treePlotter.createPlot()
myTree = treePlotter.retrieveTree(0)
#print(myTree)
#numLeaf = treePlotter.getNumLeafs(myTree)
#print(numLeaf)
#depth = treePlotter.getTreeDepth(myTree)
#print(depth)
#treePlotter.createPlot(myTree)
print(myTree)
trees.classify(myTree, labels, [0, 1])
#[1, 2, 3, 4, 5, 6]
#a = [1, 2, 3]
#b = [4, 5, 6]
#a.extend(b)
#print(a)

#[1, 2, 3, [4, 5, 6]]
#a = [1, 2, 3]
#b = [4, 5, 6]
#a.append(b)
#print(a)
