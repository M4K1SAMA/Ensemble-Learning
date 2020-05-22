import gensim
import pickle
import pandas as pd
import nltk
import sys
import os
import time
import re
import random

puncs = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"

while True:
    print(random.randint(0, 10))
    time.sleep(0.5)


train_df = pd.read_csv('train.csv', sep='\t')

sum = train_df['summary'].values
rev = train_df['reviewText'].values
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []
for i in range(len(sum)):
    sentences += sentensor.tokenize(re.sub('[%s]+'%puncs , '. ', (str)(sum[i]).lower()))
    sentences += sentensor.tokenize(re.sub('[%s]+'%puncs , '. ', (str)(rev[i]).lower()))

# print(sentences)

corpus = []

for sent in sentences:
    corpus.append(nltk.word_tokenize(sent))


w2v_model = gensim.models.Word2Vec(corpus, size=300, window=5, min_count=3, workers=4)

w2v_model.save('w2v')

# w2v = gensim.models.Word2Vec.load('w2v')
# print(w2v.most_similar('i'))
# print(w2v.similarity('i', 'my'))
# print(w2v.similarity('i', 'm3'))