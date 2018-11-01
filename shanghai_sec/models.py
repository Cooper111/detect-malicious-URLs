from keras import layers
from keras.layers import merge
from keras.models import *
from keras.layers import Embedding, Flatten, Dense, Bidirectional, LSTM
from keras.layers.core import *
#[batch_size, ax_words]
#【max_words单词， embedding_dims单词对应的维度向量】
embedding_dim = 50
rate_drop_lstm = 0.2
SINGLE_ATTENTION_VECTOR = True
EMBEDDING_DIM = 50
maxlen = 15

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps15, input_dim50)
    print(inputs.shape)
    input_dim = int(inputs.shape[2])
    time_steps = int(inputs.shape[1])
    
    a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul



# 定义模型
#【样本，每个样本100单词，每个单词100维度】

def built_model_Attention_based_LSTM(word_index,embedding_matrix):
    inputs = Input(shape=(maxlen,))
    # 层输入【samples， maxlen】
    # 层输出【samples， maxlen， dims】
    emb = Embedding(len(word_index)+1,
                        EMBEDDING_DIM,
                        input_length=maxlen,
                        weights=[embedding_matrix],
                        trainable=False,
                       )#构建词嵌入，每个单词
    emb = emb(inputs)
    attention_mul = attention_3d_block(emb)
    # model.add(layers.Conv1D(32, 5, activation='relu'))
    # model.add(layers.MaxPooling1D(3))
    x = LSTM(64, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)(emb)
    #model.add(Flatten())
    x = Dense(32, activation='relu')(x)#负责单词之间的练习和语义
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, output)
    model.summary()
    return model