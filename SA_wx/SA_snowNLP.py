#coding:utf-8
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,accuracy_score
import numpy as np
from snownlp import SnowNLP
from Data_process.testData_process import testData, testLabels,testtext

if __name__ == "__main__":

    #情感分类
    label_pre = []
    for text in testtext:
        result = SnowNLP(text)
        if result.sentiments>0.5:
            label_pre.append([0,1])
        else:
            label_pre.append([1,0])
    label = np.array(testLabels)
    labels_pre = np.array(label_pre)
    y_true = np.argmax(label, 1)
    y_predict = np.argmax(labels_pre, 1)

    # 评价
    p = precision_score(y_true, y_predict, average='macro')
    print('precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('F1:', f1)