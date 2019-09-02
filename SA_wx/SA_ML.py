# coding:utf-8
import Data_process.trainData_process as dp
import Data_process.testData_process as tp
import numpy as np
from Data_process.trainData_process import trainData, trainLabels
from Data_process.testData_process import testData, testLabels
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,accuracy_score

'''构建贝叶斯模型'''
def train_bayes(X_train, Y_train,x_test,y_test):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    Y_train=np.argmax(Y_train, 1)
    model.fit(X_train, Y_train)
    y_predict = model.predict(x_test)
    y_true = np.argmax(testLabels, 1)

    p = precision_score(y_true, y_predict, average='macro')
    print('NB_precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('NB_recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('NB_F1:', f1)

'''构建SVM模型'''
def train_SVM(X_train, Y_train,x_test,y_test):
    from sklearn.svm import SVC
    model = SVC()
    Y_train = np.argmax(Y_train, 1)
    model.fit(X_train, Y_train)
    y_predict = model.predict(x_test)
    y_true = np.argmax(testLabels, 1)

    p = precision_score(y_true, y_predict, average='macro')
    print('SVM_precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('SVM_recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('SVM_F1:', f1)

'''构建LR模型'''
def train_LR(X_train, Y_train,x_test,y_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    Y_train = np.argmax(Y_train, 1)
    model.fit(X_train, Y_train)
    y_predict = model.predict(x_test)
    y_true = np.argmax(testLabels, 1)

    p = precision_score(y_true, y_predict, average='macro')
    print('LR_precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('LR_recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('LR_F1:', f1)

'''构建XGB模型'''
def train_XGB(X_train, Y_train,x_test,y_test):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    Y_train = np.argmax(Y_train, 1)
    model.fit(X_train, Y_train)
    y_predict = model.predict(x_test)
    y_true = np.argmax(testLabels, 1)

    p = precision_score(y_true, y_predict, average='macro')
    print('XGB_precison:', p)
    r = recall_score(y_true, y_predict, average='macro')
    print('XGB_recall:', r)
    f1 = f1_score(y_true, y_predict, average='macro')
    print('XGB_F1:', f1)

if __name__ == '__main__':

    #获取训练集，shape:(文本数,句子数,embedding维度)转化为shape:(文本数,embedding维度)
    train_data = []
    for word_list in trainData:
        embedding_matrix = np.zeros(60)
        for index, word in enumerate(word_list):
            try:
                embedding_matrix += word
            except:
                pass
        train_data.append(embedding_matrix.tolist())
    train_data = np.array(train_data)
    #获取训练集标签
    train_label = trainLabels

    # 获取测试集，shape:(文本数,句子数,embedding维度)转化为shape:(文本数,embedding维度)
    test_data = []
    for word_list in testData:
        embedding_matrix = np.zeros(60)
        for index, word in enumerate(word_list):
            try:
                embedding_matrix += word
            except:
                pass
        test_data.append(embedding_matrix.tolist())
    test_data = np.array(test_data)
    # 获取测试集标签
    test_Label = np.array(testLabels)

    #机器学习模型训练
    train_bayes(train_data, train_label, test_data, test_Label)
    train_SVM(train_data, train_label, test_data, test_Label)
    train_LR(train_data, train_label, test_data, test_Label)
    train_XGB(train_data, train_label, test_data, test_Label)
