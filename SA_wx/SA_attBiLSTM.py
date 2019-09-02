# coding:utf-8
from keras.models import Sequential,Model,Input
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import LSTM,TimeDistributed,Dense, Dropout,Embedding,Bidirectional
import numpy as np
from keras.callbacks import ModelCheckpoint
from Parameters import Parameters
from Data_process.trainData_process import trainData, trainLabels
from Data_process.testData_process import testData, testLabels
from metrics import fmeasure,recall,precision

#获取参数
pm =  Parameters()

'''定义Attention Layer'''
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


'''构建BiLSTM模型，并进行迭代训练'''
def train_attBiLSTM(X_train, Y_train, X_test, Y_test):
    text_input = Input(shape=(100,60))
    #建立模型
    l_lstm = Bidirectional(LSTM(pm.hidden_dim, return_sequences=True))(text_input)
    l_dense = TimeDistributed(Dense(100))(l_lstm)
    l_att_sent = AttentionLayer()(l_dense)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model(text_input, preds)
    #定义损失函数和优化器
    model.compile(loss=pm.loss,
                  optimizer=pm.optimizer,
                  metrics=[precision,recall,fmeasure])
    filepath = "./model/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_fmeasure', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    #训练
    model.fit(X_train, Y_train, batch_size=pm.batch_size,callbacks=callbacks_list, epochs=pm.epochs, validation_data=(X_test, Y_test))


if __name__ == '__main__':

    # 打印训练集和测试集形状
    print("The trainData's shape is:", trainData.shape)
    print("The testData's shape is:", testData.shape)
    print("The trainLabels's shape is:", trainLabels.shape)
    print("The testLabels's shape is:", testLabels.shape)

    # 训练attBiLSTM模型
    train_attBiLSTM(trainData, trainLabels, testData, testLabels)