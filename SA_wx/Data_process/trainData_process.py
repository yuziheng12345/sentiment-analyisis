import jieba
from Parameters import Parameters
import numpy as np
from random import randint
from random import shuffle
import gensim
from gensim.models import KeyedVectors
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec

pm = Parameters()

# model = KeyedVectors.load_word2vec_format('model/sgns.weibo.word')
model = gensim.models.Word2Vec.load('model/Word60.model')

'''获取停用词表'''
def makeStopWord():
    with open(pm.stop_word,'r',encoding = 'utf-8') as f:
        lines = f.read()
        words = jieba.lcut(lines,cut_all = False)
        stopWord =[]
        for word in words:
            stopWord.append(word)
    return stopWord

'''所有词语转化为向量'''
def words2Array(lineList):
    linesArray=[]
    wordsArray=[]
    for j in range(len(lineList)):
        line = lineList[j]
        t = 0
        p = 0
        for i in range(100):
            if i<len(line):
                try:
                    wordsArray.append(model.wv.word_vec(line[i]))
                    p = p + 1
                except KeyError:
                    t = t + 1
                    continue
            else:
               wordsArray.append(np.array([0.0]*60))
        for i in range(t):
            wordsArray.append(np.array([0.0]*60))
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    return linesArray

'''打包解包，随机数据顺序'''
def convert2Data1(posArray,negArray):
    randIt = []
    data = []
    labels = []
    for i in range(len(posArray)):
        randIt.append([posArray[i], [0,1]])
    for i in range(len(negArray)):
        randIt.append([negArray[i], [1,0]])
    shuffle(randIt)
    for i in range(len(randIt)):
        data.append(randIt[i][0])
        labels.append(randIt[i][1])
    data = np.array(data)
    return data, labels

'''句子转化为词组'''
def getWords(file):
    stopWord = makeStopWord()
    wordList = []
    trans = []
    lineList = []
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        trans = jieba.lcut(line.replace('\n',''), cut_all = False)
        for word in trans:
            if word not in stopWord:
                wordList.append(word)
        lineList.append(wordList)
        wordList = []
    return lineList

'''制作训练集'''
def make_traindata(posPath,negPath):
    #获取词汇，返回类型为[[word1,word2...],[word1,word2...],...]
    pos = getWords(posPath)
    neg = getWords(negPath)
    #将评价数据转换为矩阵，返回类型为array
    posArray = words2Array(pos)
    negArray = words2Array(neg)
    #将积极数据和消极数据混合在一起打乱，制作数据集
    Data,  Labels = convert2Data1(posArray,negArray)
    Data = np.array(Data)
    Labels = np.array(Labels)
    return Data, Labels


trainData, trainLabels = make_traindata(pm.train_posdata,pm.train_negdata)