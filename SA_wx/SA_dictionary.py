#coding=utf-8
import numpy as np
from Parameters import  Parameters
import Data_process.testData_process as dp
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,accuracy_score

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
    return pos_score,neg_score



if __name__ == '__main__':

    # 读取参数
    pm = Parameters()

    # 读取各个词典返回列表
    deny_word = open_dict(Dict=pm.deny_word)
    posdict = open_dict(Dict=pm.posdict)
    negdict = open_dict(Dict=pm.negdict)
    degree_word = open_dict(Dict=pm.degree_word)
    link_word = open_dict(Dict=pm.link_word)
    dj_word = open_dict(Dict=pm.dj_word)
    mostdict = degree_word[degree_word.index('extreme') + 1: degree_word.index('very')]  # 权重4，即在情感词前乘以4
    verydict = degree_word[degree_word.index('very') + 1: degree_word.index('more')]  # 权重3
    moredict = degree_word[degree_word.index('more') + 1: degree_word.index('ish')]  # 权重2
    ishdict = degree_word[degree_word.index('ish') + 1: degree_word.index('last')]  # 权重0.5

    # 获取测试集的内容和标签
    contents, labels, text_list = dp.read_testdata_dic(pm.test_data)

    #情感分类
    labels_pre = []
    for content in contents:
        pos_score, neg_score = score(seg_score(content))
        if pos_score >= neg_score:
            labels_pre.append([0,1])
        else:
            labels_pre.append([1,0])
    labels = np.array(labels)
    labels_pre = np.array(labels_pre)
    y_true = np.argmax(labels, 1)
    y_predict = np.argmax(labels_pre, 1)

    # 评价
    p = precision_score(y_true, y_predict, average='macro')
    print('precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('F1:', f1)
