# sentiment analyisis for wechat articles

微信文章情感分析项目，对比模型：1.深度学习模型：Attention-BiLSTM、BiLSTM、LSTM、CNN;2.机器学习模型：SVM、NB、XGBoost、LR；3.开源情感分析工具：Senta、SnowNLP

1.文件说明

data文件夹:

     jd-train: 京东开源训练集
     sogou-train: 搜狗训练集
     wx-train：微信训练集
     Worddict：词典       
	test_wx.txt：微信文章测试集
    停用词.txt：停用词
	
Data_process文件夹：数据处理

	samples：搜狗新闻数据集，经过情感标注后得到搜狗训练集
	tag_trainData.py：情感词典标注法筛选训练集和标注标签
	testData_process.py：测试集处理
	trainData_process.py：训练集处理
	weixin_info.txt：微信分类数据集，经过情感标注后得到微信训练集
	
model文件夹：

	sgns.weibo.bigram 预训练的300d词向量
	sgns.weibo.word    
	weights.best.hdf5 保存的att-bilstm模型
	Word60.model 预训练的60d词向量
	Word60.model.syn0.npy
	Word60.model.syn1neg.npy
	
GUI.py 可视化界面

metrics.py 评价指标

Parameters.py 参数设置

SA_attBiLSTM.py attBiLSTM模型

SA_BiLSTM.py BiLSTM模型

SA_CNN.py CNN模型

SA_dictionary.py 词典方法

SA_LSTM.py LSTM模型

SA_ML.py 机器学习方法

SA_Senta.py senta工具

SA_snowNLP.py snowNLP工具

2.需要安装的第三方包：
sklearn，gensim，keras，numpy，tkinter，xgboost，paddlehub，snownlp

