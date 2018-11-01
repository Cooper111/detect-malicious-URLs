from keras import layers
from keras.layers import merge
from keras.models import *
from keras.layers import Embedding, Flatten, Dense, Bidirectional, LSTM
from keras.layers.core import *
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from load import load_data
from preprocess.data_split import pipeline_x
from preprocess.data_tokenizer import get_tokenizer
from embedding import get_embedding_weights
from utils import get_available_gpus, ensure_folder, get_best_model, get_highest_acc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from models import attention_3d_block
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

def built_model_Attention_based_LSTM(word_index):
	# 层输入【samples， maxlen】
	# 层输出【samples， maxlen， dims】
	maxlen = 15
	EMBEDDING_DIM = 50
	word_index = word_index
	embedding_dim = 50
	rate_drop_lstm = 0.2
	SINGLE_ATTENTION_VECTOR = True
	EMBEDDING_DIM = 50
	maxlen = 15
	inputs = Input(shape=(maxlen,))
	emb = Embedding(len(word_index)+1,
					EMBEDDING_DIM,
					input_length=maxlen,
					#weights=[embedding_matrix],
					trainable=False,
					)#构建词嵌入，每个单词
	emb = emb(inputs)
	attention_mul = attention_3d_block(emb)
	x = LSTM(64, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)(emb)
	x = Dense(32, activation='relu')(x)#负责单词之间的练习和语义
	output = Dense(1, activation='sigmoid')(x)
	model = Model(inputs, output)
	model.summary()
	return model

if __name__ == '__main__':
	#ensure_folder('models')
	best_model, epoch = get_best_model()
	if best_model is None:
		initial_epoch = 0
	else:
		initial_epoch = epoch + 1

	
	#==================================================
	#处理test.csv
	test= pd.read_csv('data/test.csv', encoding='utf-8')
	test_x = test.iloc[:,0]
	pre_data = test_x.apply(lambda x:pipeline_x(x))
	maxlen = 15
	testing_sample = pre_data.shape[0] 
	max_words = 10000

	f=open('./word_index.pk','rb')
	word_index=pickle.load(f)
	f.close()

	temp = []
	temp_list = []
	for i in pre_data:
		t = i.strip().split()
		for j in t:
			if j  in word_index.keys():
				temp.append(word_index[j])
		temp_list.append(temp)

	sequences = temp_list
	#print(sequences[0])

	#sequences = tokenizer.texts_to_sequences(pre_data)#如果model=‘bianry’则是01表示


	data = pad_sequences(sequences, maxlen=maxlen)#maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
	
	embedding_matrix = get_embedding_weights(word_index)

	model = built_model_Attention_based_LSTM(word_index)
	model.load_weights(best_model)
	model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])
	temp = model.predict(data)

	result_dir = './result/'

	result_list = []
	for i in temp:
		t = int(i)
		if t >= 0.5:
			t = 1
		else:
			t = 0
		result_list.append(t)

	with open(result_dir+'submit.json', 'w') as f:
		#f.write(str(result))
		json.dump(np.array(result_list).tolist(), f,ensure_ascii=False)
		print('write result json, num is %d' % len(result_list))

