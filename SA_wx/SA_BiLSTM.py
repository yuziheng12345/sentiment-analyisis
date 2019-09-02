# coding:utf-8
from keras.models import Sequential
from keras import backend as K
from keras.layers import LSTM,TimeDistributed,Dense, Dropout,Embedding,Bidirectional
from Parameters import  Parameters
import numpy as np
from Data_process.trainData_process import trainData, trainLabels
from Data_process.testData_process import testData, testLabels
from metrics import fmeasure,recall,precision

#获取参数
pm =  Parameters()

'''构建BiLSTM模型，并进行迭代训练'''
def train_BiLSTM(X_train, Y_train, X_test, Y_test):

    model = Sequential()
    #构建Bilstm层，隐藏层节点数为hidden_dim，文本长度为100，embedding为60
    model.add(Bidirectional(LSTM(pm.hidden_dim,return_sequences=False),input_shape=(100,60),merge_mode='concat'))
    #dropout
    model.add(Dropout(pm.dropout))
    #全连接层
    model.add(Dense(2, activation='softmax'))
    #定义损失函数和优化器
    model.compile(loss=pm.loss,
                  optimizer=pm.optimizer,
                  metrics=[precision,recall,fmeasure])
    #训练
    model.fit(X_train, Y_train, batch_size=pm.batch_size, epochs=pm.epochs, validation_data=(X_test, Y_test))


if __name__ == '__main__':

    # 打印训练集和测试集形状
    print("The trainData's shape is:", trainData.shape)
    print("The testData's shape is:", testData.shape)
    print("The trainLabels's shape is:", trainLabels.shape)
    print("The testLabels's shape is:", testLabels.shape)

    # 训练BiLSTM模型
    train_BiLSTM(trainData, trainLabels, testData, testLabels)