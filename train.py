import pandas as pd
from sklearn import tree
import random
import pickle
import gensim
import nltk
import re
import os

bagging = 10

puncs = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"

w2v = gensim.models.Word2Vec.load('w2v')

def load_data():
    X = []

    test_X = []
    if 'train_X' in os.listdir('.'):
        with open('train_X', 'rb') as f:
            X = pickle.load(f)
    else:
        train_df = pd.read_csv('train.csv', sep='\t')
        sum = train_df['summary'].values
        rev = train_df['reviewText'].values
        sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        for i in range(len(sum)):
            sentences += sentensor.tokenize(re.sub('[%s]' % puncs, '. ', (str)(sum[i]).lower())) + ' ' + sentensor.tokenize(re.sub('[%s]' % puncs, '. ', (str)(rev[i]).lower()))

        for sent in sentences:
            X.append(nltk.word_tokenize(sent))
        with open('train_X', 'wb') as f:
            pickle.dump(X, f)

    if 'train_Y' in os.listdir('.'):
        with open('train_Y', 'rb') as f:
            Y = pickle.load(f)
    else:
        train_df = pd.read_csv('train.csv', sep='\t')
        Y = train_df['overall'].values
        with open('train_Y', 'wb') as f:
            pickle.dump(Y, f)

    if 'test_X' in os.listdir('.'):
        with open('test_X', 'rb') as f:
            test_X = pickle.load(f)
    else:
        test_df = pd.read_csv('test.csv', sep='\t')
        sum = test_df['summary'].values
        rev = test_df['reviewText'].values
        sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = []
        for i in range(len(sum)):
            sentences += sentensor.tokenize(re.sub('[%s]' % puncs, '. ', (str)(sum[i]).lower()))
            sentences += sentensor.tokenize(re.sub('[%s]' % puncs, '. ', (str)(rev[i]).lower()))
        for sent in sentences:
            test_X.append(nltk.word_tokenize(sent))
        with open('test_X', 'wb') as f:
            pickle.dump(test_X, f)
    return X, Y, test_X


def process(X):
    ret = []
    if 'product' in os.listdir('.'):
        with open('product', 'rb') as f:
            ret = pickle.load(f)
    else:
        for i in X:
            tmp = [0, 0, 0]
            for word in i:
                try:
                    positive = w2v.similarity('good', word)
                    negative = w2v.similarity('bad', word)
                    nt = w2v.similarity('not', word)
                    if positive > negative and positive > nt:
                        tmp[0] += positive
                    elif negative > positive and negative > nt:
                        tmp[1] += negative
                    elif nt > positive and nt > negative:
                        tmp[2] += nt
                except KeyError:
                    pass
            ret.append(tmp)
        with open('product', 'wb') as f:
            pickle.dump(ret, f)
    return ret


def train(X, Y):
    clf = tree.DecisionTreeClassifier()
    X = process(X)
    for i in range(bagging):
        print('training %i'%i)
        subX = []
        subY = []
        for j in range(len(X)):
            rand = random.randint(0, len(X) - 1)
            subX.append(X[rand])
            subY.append(Y[rand])
        clf.fit(subX, subY)
        print("wrote %i"%i)
        pickle.dump('models/DTree_%i.pickle'%i)


def test(X):
    ret = []
    clfs = []
    X = process(X)
    for i in range(bagging):
        clfs.append(pickle.load('models/DTree_%i.pickle'%i))
    for i in len(X):
        tmp = []
        for clf in clfs:
            res = clf.predict(X[i])
            tmp.append(res)
        ret.append(tmp)
    return ret

