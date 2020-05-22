import pandas as pd
from gensim import corpora, models
import pickle
import string
import re

puncs = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"


train_df = pd.read_csv('train.csv', sep='\t')

corpus = []

# sum = train_df['summary'].values
# rev = train_df['reviewText'].values
# length = len(sum)
# corpus = [(str)(sum[i]) + " " + (str)(rev[i]) for i in range(length)]
# with open('corpus.pickle', 'wb') as f:
#     pickle.dump(corpus, f)

with open('corpus.pickle', 'rb') as f:
    corpus = pickle.load(f)
# print(corpus)
# for text in corpus:
#     print(text)
word_list = []
for i in range(len(corpus)):
    tmp = re.split(' ', re.sub(r'[%s]+' % puncs, ' ', corpus[i]))
    if '' in tmp:
        tmp.remove('')
    word_list.append(tmp)

dictionary = ''
# dictionary = corpora.Dictionary(word_list)
# with open('dict.pickle', 'wb') as f:
#     pickle.dump(dictionary, f)
with open('dict.pickle', 'rb') as f:
    dictionary = pickle.load(f)
# print(type(dictionary))
# print(len(dictionary))
new_corpus = [dictionary.doc2bow(text) for text in word_list]


# tfidf = models.TfidfModel.load('tfidf.model')
tfidf = models.TfidfModel(new_corpus)
tfidf.save('tfidf.model')

tf = {}
for i in dictionary.values():
    print(tfidf[dictionary.doc2bow([i])])
