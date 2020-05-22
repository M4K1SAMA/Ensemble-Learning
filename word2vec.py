import gensim
import pickle
import nltk

import gensim

model = gensim.models.Word2Vec.load('w2v_200')

print(model.similarity('good', 'bad'))

print(model.most_similar('tadwhbdwj'))

