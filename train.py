import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import random
import pickle

bagging = 10


def bow(inputs):
    cv = CountVectorizer()
    cv.fit(inputs)
    vector = cv.transform((inputs))
    return vector.toarray()


def load_data():
    train_df = pd.read_csv('train.csv', sep='\t')
    test_df = pd.read_csv('test.csv', sep='\t')

    sum = train_df['summary'].values
    rev = train_df['reviewText'].values
    length = len(sum)


    X = np.array([(str)(sum[i]) + " " + (str)(rev[i]) for i in range(length)])
    Y = train_df['overall'].values
    sum = test_df['summary'].values
    rev = test_df['reviewText'].values
    length = len(sum)
    test_X = np.array([(str)(sum[i]) + " " + (str)(rev[i]) for i in range(length)])
    return X, Y, test_X


def train(X, Y):
    clf = tree.DecisionTreeClassifier()
    X = bow(X)
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
    X = bow(X)
    for i in range(bagging):
        clfs.append(pickle.load('models/DTree_%i.pickle'%i))
    for i in len(X):
        tmp = []
        for clf in clfs:
            res = clf.predict(X[i])
            tmp.append((res))
        ret.append((tmp))
    return ret
