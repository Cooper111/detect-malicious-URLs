import os
import numpy as np
def get_embedding_weights(word_index):
	GLOVE_DIR = 'D:\Jupyter\glove.6B'
	embeddings_index = {}
	with open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding='utf-8') as f:
	    for line in f:
	        values = line.split()
	        word = values[0]
	        coefs = np.asarray(values[1:], dtype='float32')
	        embeddings_index[word] = coefs

	print('Found %s word vectors.' % len(embeddings_index))

	EMBEDDING_DIM = 50
	embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	return embedding_matrix

