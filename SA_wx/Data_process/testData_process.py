# -*- coding:utf-8 -*-
import jieba
from Parameters import Parameters
import re
import xlwt
import numpy as np
from gensim.models import KeyedVectors
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim

pm =  Parameters()

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

'''所有词语转化为向量(文本数,句子数,embedding)'''
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

'''制作测试数据'''
def read_testdata(filename=pm.test_data):
    '''
    内容分句、分词，将内容与标签分开，存入列表
    '''
    stopWord = makeStopWord()
    with open(filename, encoding='utf-8') as file:
        contents, labels, textList = [], [], []
        texts = file.read()
        text_list = texts.split('\n')
        for text in text_list:
            [tag_, char] = text.strip().split('\t', 1)
            if tag_=='正面':
                labels.append([0,1])
            else:
                labels.append([1,0])
            textList.append(char)
            trans = jieba.lcut(char, cut_all=False)
            wordList=[]
            for word in trans:
                if word not in stopWord and word != '\t':
                    wordList.append(word)
            if wordList:
                    contents.append(wordList)
    return contents,labels,textList

'''制作词典方法的测试数据'''
def read_testdata_dic(filename=pm.test_data):
    stopWord = makeStopWord()
    with open(filename, encoding='utf-8') as file:
        contents, labels, textList = [], [], []
        texts = file.read()
        text_list = texts.split('\n')
        for text in text_list:
            [tag_, char] = text.strip().split('\t', 1)
            if tag_=='正面':
                labels.append([0,1])
            else:
                labels.append([1,0])
            textList.append(char)
            sentences = re.split('(。|！|\!|\.|？|\?)', char)
            content_list = []
            for sentence in sentences:
                wordList = []
                trans = jieba.lcut(sentence, cut_all=False)
                for word in trans:
                    if word not in stopWord and word != '\t':
                        wordList.append(word)
                if wordList:
                    content_list.append(wordList)
            contents.append(content_list)
    return contents,labels,textList

'''制作GUI的测试数据'''
def read_testdata_gui(char):
    '''
    内容分句、分词，将内容与标签分开，存入列表
    '''
    stopWord = makeStopWord()
    trans = jieba.lcut(char, cut_all=False)
    wordList=[]
    for word in trans:
        if word not in stopWord and word != '\t':
                wordList.append(word)
    t = 0
    p = 0
    wordsArray = []
    for i in range(100):
        if i < len(wordList):
            try:
                wordsArray.append(model.wv.word_vec(wordList[i]).tolist())
                p = p + 1
            except KeyError:
                t = t + 1
                continue
        else:
            wordsArray.append([0.0] * 60)
    for i in range(t):
        wordsArray.append([0.0] * 60)
    return np.array([wordsArray])

'''输出结果保存到Excel'''
#para1为文本列表
def write_excel(textlist,y_true, y_predict,filename):
    workbook = xlwt.Workbook(encoding='UTF-8')
    worksheet = workbook.add_sheet('My Worksheet')
    worksheet.write(0, 0, label='文本')
    worksheet.write(0, 1, label='真实标签')
    worksheet.write(0, 2, label='预测标签')
    z = 0
    for i in range(len(textlist)):
        z = z + 1
        # 将x和y输出到表格，x是某一行的词，y是某一行的极性
        worksheet.write(z, 0, label= textlist[i])
        worksheet.write(z, 1, label='%s'%(y_true[i]-1))
        worksheet.write(z, 2, label='%s'%(y_predict[i]-1))
    workbook.save(filename)

# 获取测试数据，标签，文本列表
testData, testLabels, testtext = read_testdata(pm.test_data)
testData = words2Array(testData)
testLabels = np.array(testLabels)
testtext = np.array(testtext)