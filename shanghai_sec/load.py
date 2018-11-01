import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
	#test = pd.read_csv('test.csv',encoding='utf-8')
	train= pd.read_csv('data/train.csv', encoding='utf-8')

	train_x = train.iloc[:,0]
	train_y = train.iloc[:,1]


	lbl = LabelEncoder()
	train_y = lbl.fit_transform(train_y)
	return train_x, train_y