import gensim
import pickle
import nltk

import gensim

sentensor = nltk.data.load('tokenizers/punkt/english.pickle')

print(nltk.word_tokenize('dwa. daww'))

print(sentensor.tokenize('dwa. dw'))