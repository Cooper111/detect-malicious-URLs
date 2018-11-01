#处理文本为序列的整数列表，制作x，y
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def get_tokenizer(pre_data):
	maxlen = 15
	training_sample = pre_data.shape[0]
	#validation_samples = 
	max_words = 10000
	#max_words个单词的文本转数值转换器
	tokenizer = Tokenizer(num_words=max_words, lower=False)
	tokenizer.fit_on_texts(pre_data)
	#将字符串转化为整数索引的列表
	sequences = tokenizer.texts_to_sequences(pre_data)#如果model=‘bianry’则是01表示
	#获得  字符：数字 的字典
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	#将列表转换为（samples，maxlen）的二维整数张量
	data = pad_sequences(sequences, maxlen=maxlen)#maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
	print('Shape of data tensor:', data.shape)
	return data,word_index,tokenizer