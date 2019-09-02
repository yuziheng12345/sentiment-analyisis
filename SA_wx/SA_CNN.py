# coding:utf-8
import numpy as np
from keras.models import Model,Input
from keras.layers import Dense, Dropout,Flatten,Embedding,MaxPooling1D,Convolution1D
from keras.layers.merge import Concatenate
from Parameters import  Parameters
from Data_process.trainData_process import trainData, trainLabels
from Data_process.testData_process import testData, testLabels
from metrics import fmeasure,recall,precision

#获取参数
pm =  Parameters()

'''构建CNN模型，并进行迭代训练'''
def train_cnn(X_train, Y_train, X_test, Y_test):
    model_input = Input(shape=(100, 60))
    kernel_sizes = pm.kernel_sizes
    conv_blocks=[]
    for sz in kernel_sizes:
        conv = Convolution1D(filters=pm.filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=pm.strides)(model_input)
        conv = MaxPooling1D()(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)
    model_output = Dense(2, activation="sigmoid")(z)
    model = Model(model_input, model_output)
    #定义损失函数和优化器
    model.compile(loss=pm.loss,
                  optimizer='adam',   metrics = [precision, recall,fmeasure])
    #训练
    model.fit(X_train, Y_train, batch_size=pm.batch_size, epochs=pm.epochs, validation_data=(X_test, Y_test))


if __name__ == '__main__':

    #打印训练集和测试集形状
    print("The trainData's shape is:", trainData.shape)
    print("The testData's shape is:", testData.shape)
    print("The trainLabels's shape is:", trainLabels.shape)
    print("The testLabels's shape is:", testLabels.shape)

    #训练CNN模型
    train_cnn(trainData, trainLabels, testData, testLabels)


