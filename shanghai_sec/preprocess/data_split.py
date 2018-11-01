import pandas as pd
import numpy as np
import nltk

#先'/'再'-'再'.'再'com'
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')	#get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')	#get tokens after splitting by dash
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(allTokens)#-remove_set--#remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return allTokens

#进过GetTokens后在使用nltk进一步分词，主要为了加强对特殊符号的处理
def preprocessing_data_2(list_data):
    new_data = ''
    for i in list_data:
        words = nltk.tokenize.TweetTokenizer(strip_handles=False, reduce_len = True).tokenize(i)
        temp = ' '.join(words)
        new_data = new_data + ' '+temp
    return new_data

#制成Pipeline
def pipeline_x(data):
    data = getTokens(data)
    data = preprocessing_data_2(data)
    return data

