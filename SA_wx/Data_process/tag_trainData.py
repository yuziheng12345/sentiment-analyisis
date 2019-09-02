import jieba
import numpy as np
import os
import sklearn
from sklearn.naive_bayes import MultinomialNB
import re

'''打开词典文件，返回列表'''
def open_dict(Dict):
    path = '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

'''数量奇偶判断'''
def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'

'''给各个分句打分'''
def seg_score(sentences):
     seg_sentence = sentences
     count1 = []#[积极分数，消极分数]
     count2 = []#[[积极分数，消极分数],[],...]
     for words in seg_sentence: #循环遍历每一分句
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        pos_sum = 0 #记录积极词数目
        neg_sum = 0 #记录消极词数目
        pos_score = 0#积极词总分
        neg_score = 0#消极词总分
        for segtmp in words:
            if segtmp in negdict:  # 判断词语是否是消极情感词
                negcount = 1
                d = 0
                for w in segtmp[a:i]:# 扫描情感词前的程度词
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':# 扫描情感词前的否定词数
                    negcount *= -1.0
                else:
                    pass
                neg_sum += 1
                a = i + 1
                neg_score += negcount

            elif segtmp in posdict:  # 积极情感的分析，与上面一致
                poscount = 1
                c = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1
                if judgeodd(c) == 'odd':
                    poscount *= -1.0
                else:
                    pass
                pos_sum += 1
                a = i + 1
                pos_score  += poscount

            elif segtmp == '！' or segtmp == '!':  ##判断句子是否有感叹号
                for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict :
                        pos_score += 2
                        break
                    elif w2 in posdict :
                        neg_score += 2
                        break
            i += 1 # 扫描词位置前移

        # 以下是防止出现负数的情况
        if pos_score < 0 and neg_score > 0:
                neg_score = neg_score-pos_score
        elif neg_score < 0 and pos_score > 0:
                pos_score = pos_score-neg_score
        elif pos_score < 0 and neg_score < 0:
                pos_score = -neg_score
                neg_score = -pos_score
        else:
                pass

        if  segtmp != []:
            if segtmp[0] in link_word:
                count1.extend((pos_score*1.5,neg_score*1.5))
            elif segtmp[0] in dj_word:
                count1.extend((pos_score*2,neg_score*2))
            else:
                count1.extend((pos_score,neg_score))
        count2.append(count1)
        count1 = []
     return count2

'''给整个文本打分'''
def score(seg_score):
    pos_score = 0
    neg_score = 0
    for i in range(len(seg_score)):
        if seg_score[i] != []:
            pos_score += seg_score[i][0]
            neg_score += seg_score[i][1]
    return pos_score/len(seg_score),neg_score/len(seg_score)

'''获取停用词表'''
def makeStopWord(stopwords_file):
    with open(stopwords_file,'r',encoding = 'utf-8') as f:
        lines = f.read()
        words = jieba.lcut(lines,cut_all = False)
        stopWord =[]
        for word in words:
            stopWord.append(word)
    return stopWord

'''
文本预处理
返回值： 
contents：[[词语1，词语2]，[],[]...]
textList：[文本1，文本2,...]
'''
def TextProcessing(folder_path, stopWord):
        files = os.listdir(folder_path)
        # 类内循环
        contents = []
        textList=[]
        for file in files:
            print(file)
            new_folder = os.path.join(folder_path, file)
            with open(new_folder,'r',encoding='utf-8') as fp:
                   texts = fp.readlines()
                   for char in texts:
                       textList.append(char)
                       sentences = re.split('(。|！|\!|\.|？|\?)', char)
                       content_list = []
                       for sentence in sentences:
                           wordList = []
                           trans = jieba.lcut(sentence, cut_all=False)
                           for word in trans:
                               if word not in stopWord and word != '\t' and word !='\n' :
                                   wordList.append(word)
                           if wordList:
                               content_list.append(wordList)
                       contents.append(content_list)
        return contents,textList

if __name__ == '__main__':

    # 读取各个词典返回列表
    deny_word = open_dict(Dict = '../data/Worddict/否定词')
    posdict = open_dict(Dict = '../data/Worddict/positive')
    negdict = open_dict(Dict='../data/Worddict/negative')
    degree_word = open_dict(Dict='../data/Worddict/程度级别词语')
    link_word = open_dict(Dict='../data/Worddict/转折连词')
    dj_word = open_dict(Dict='../data/Worddict/递进连词')
    mostdict = degree_word[degree_word.index('extreme') + 1: degree_word.index('very')]  # 权重4，即在情感词前乘以4
    verydict = degree_word[degree_word.index('very') + 1: degree_word.index('more')]  # 权重3
    moredict = degree_word[degree_word.index('more') + 1: degree_word.index('ish')]  # 权重2
    ishdict = degree_word[degree_word.index('ish') + 1: degree_word.index('last')]  # 权重0.5
    stopwords_file = '../data/停用词.txt'
    stopwords_set = makeStopWord(stopwords_file)

    # 文本预处理
    folder_path = './samples'
    contents,textList = TextProcessing(folder_path,stopwords_set)

    # 对文本进行情感打分
    sc_list=[]
    for content in contents:
        pos_score, neg_score = score(seg_score(content))
        sc = pos_score - neg_score
        sc_list.append(sc)

    # 将情感分值压缩在-1到1
    max = max(sc_list)
    score_list = []
    for sc in sc_list:
        sc = sc / max
        score_list.append(sc)

    # 正向阈值和负向阈值
    pos_limit=0.01
    neg_limit=-0.01

    pos_contents,neg_contents=[],[]
    train_labels = []
    print('文本数：',len(contents))

    # 筛选训练集并标注训练集标签
    for i in range(len(contents)):
        if score_list[i]>pos_limit:
            train_labels.append([0,1])
            pos_contents.append(textList[i])
        elif score_list[i]<neg_limit:
            train_labels.append([1,0])
            neg_contents.append(textList[i])

    print('正面文本数：',len(pos_contents))
    for x in pos_contents:
        file = 'pos_data' + '.txt'
        f = open(file,'a+',encoding='utf-8')
        f.write(x)

    print('负面文本数：',len(neg_contents))
    for x in neg_contents:
        file = 'neg_data' + '.txt'
        f = open(file,'a+',encoding='utf-8')
        f.write(x)
