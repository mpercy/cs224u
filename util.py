from scipy.spatial import distance
import sklearn.metrics.pairwise
import inspect
import logging
import os.path
import re
import sys
from math import isnan

import itertools
import gensim
import numpy as np

sentenceEnds = ['...', '.', ';', '!', '?']
sentenceEndPattern = re.compile('|'.join([re.escape(tok) for tok in sentenceEnds]))

def sentenceSeg(doc):
    # new paragraph is meaningless here
    doc = re.sub(r'\s+', ' ', doc)
    # split the doc with sentence ending marks
    initialRegions = re.split(sentenceEndPattern, doc)
    return [x for x in initialRegions if x != '']

def cosine(x, y):
    rlt =  distance.cosine(x, y)
    if isnan(rlt):
        # TODO: what's the best
        return -2
    return rlt

def cosine_sparse(x, y):
    return sklearn.metrics.pairwise.cosine_similarity(x, y)

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def isEmpty():
        return self._queue==[]

class DataSet(gensim.utils.SaveLoad):
    """
    Uses the gensim mmap stuff to efficiently save/load a data set.
    To get the efficiency benefit, the given arguments should be
    scipy.sparse.csr_matrix instances or numpy ndarrays.
    """
    def __init__(self, train, trainY, test, testY):
        self.train = train
        self.trainY = trainY
        self.test = test
        self.testY = testY
