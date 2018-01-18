#P67
from numpy import *
import re
import operator

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please' ],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #print("document: ", document)
        vocabSet = vocabSet | set(document) #创建两个集合的并集
        #print("vocabSet: ", vocabSet)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    print("trainMatrix: ", trainMatrix)
    print("trainCategory: ", trainCategory)
    numTrainDocs = len(trainMatrix)
    print("numTrainDocs: ", numTrainDocs)
    numWords = len(trainMatrix[0])
    print("numWords: ", numWords)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #p0Num = zeros(numWords)
    p0Num = ones(numWords)
    #print("p0Num: ", p0Num)
    #p1Num = zeros(numWords)
    p1Num = ones(numWords)
    #print("p1Num: ", p1Num)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            print("trainMatrix[i]: ", trainMatrix[i])
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print("p1Num: ", p1Num)
    print("p1Denom: ", p1Denom)
    print("p0Num: ", p0Num)
    print("p0Denom: ", p0Denom)
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

"""
vec2Claasify: 待分类的词条数组
p0Vec: 侮辱类分类概率
p1Vec: 非侮辱类分类概率
pClass1： 文档属于侮辱类的概率
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

#基于词袋模型，词袋中每个单词可以出现多次。跟函数setOfWords2Vec()的区别是，+=1，而不是=1
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        print(i)
        wordList = textParse(open(f'email/spam/{i}.txt').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(f'email/ham/{i}.txt').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    print("init trainingSet: ", trainingSet)
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        print(randIndex)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: {}".format(float(errorCount)/len(testSet)))

#返回出现频率最高的30个词
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        #print("token: ", token)
        freqDict[token] = fullText.count(token)
    #operator.itemgetter(1) 获取对象第一个域（第二列）的值
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]
#RSS 源分类器
def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print(f"the error rate is: {float(errorCount)/len(testSet)}")
            return vocabList, p0V, p1V

#显示最具表征性的词汇
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print(10 * "SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print(10 * "NY**")
    for item in sortedNY:
        print(item[0])
