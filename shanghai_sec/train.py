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
from models import built_model_Attention_based_LSTM

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
	ensure_folder('models')
	best_model, epoch = get_best_model()
	if best_model is None:
		initial_epoch = 0
	else:
		initial_epoch = epoch + 1

	train_x, train_y = load_data()
	#============分词======================
	pre_data = train_x.apply(lambda x:pipeline_x(x))
	#============Tokenizer==================
	data, word_index,tokenizer = get_tokenizer(pre_data)

	#=====对数据进行分割,先打乱=============
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = train_y[indices]

	train_sample = data.shape[0]  - (data.shape[0] // 10)
	print('sum',data.shape[0])
	print('indexs',train_sample)

	x_train = data[:train_sample]
	y_train = labels[:train_sample]
	x_val = data[train_sample:]
	y_val = labels[train_sample:]

	#========加载词向量=============
	embedding_matrix = get_embedding_weights(word_index)

	#========构建Callback===========
	class MyCbk(keras.callbacks.Callback):
		def __init__(self, model):
			keras.callbacks.Callback.__init__(self)
			self.model_to_save = model

		def on_epoch_end(self, epoch, logs=None):
			fmt = 'models/model.%02d-%.4f.hdf5'
			highest_acc = get_highest_acc()
			if float(logs['val_acc']) > highest_acc:
				self.model_to_save.save(fmt % (epoch, logs['val_acc']))


#=============== Callbacks ====================
	tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
	early_stop = EarlyStopping('val_acc', patience=5)
	reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5, patience=int( 5 / 4), verbose=1)
	trained_models_path = 'models/model'
	model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'
	model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
#=================================================
	# num_gpu = 1  # fix
	# if num_gpu >= 2:
	# 	with tf.device("/cpu:0"):
	# 		model = built_model_Attention_based_LSTM(word_index,embedding_matrix)
	# 		if best_model is not None:
	# 			model.load_weights(best_model)
	
	# 	model = multi_gpu_model(model, gpus=num_gpu)
	# 	# rewrite the callback: saving through the original model and not the multi-gpu model.
	# 	model_checkpoint = MyCbk(model)
	# else:
	# 	model = built_model_Attention_based_LSTM(word_index,embedding_matrix)
	# 	if best_model is not None:
	# 		model.load_weights(best_model)
	model = built_model_Attention_based_LSTM(word_index,embedding_matrix)
	model.load_weights(best_model)
	#====================================================
	callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
	#=================================================
	model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])
	history = model.fit(x_train, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_data=(x_val, y_val),
                   initial_epoch=initial_epoch,
                   callbacks=callbacks,
                   verbose =1)




