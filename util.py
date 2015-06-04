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

class SimpleDict(gensim.utils.SaveLoad):
    def __init__(self):
        self.token2id = {}
        self.id2token = []
        self.vectors = []

    def __len__(self):
        return len(self.id2token)

    def finalize(self):
        self.vectors = np.vstack(self.vectors)

    # Pilfered from gensim.corpora.Dictionary
    def doc2bow(self, document):
        """
        Convert `document` (a list of words) into the bag-of-words format = list
        of `(token_id, token_count)` 2-tuples. Each word is assumed to be a
        **tokenized and normalized** string (either unicode or utf8-encoded). No further preprocessing
        is done on the words in `document`; apply tokenization, stemming etc. before
        calling this method.
        """
        result = {}
        document = sorted(gensim.utils.to_unicode(token) for token in document)
        # construct (word, frequency) mapping. in python3 this is done simply
        # using Counter(), but here i use itertools.groupby() for the job
        for word_norm, group in itertools.groupby(document):
            frequency = len(list(group)) # how many times does this word appear in the input document
            tokenid = self.token2id.get(word_norm, None)
            if tokenid is None:
                continue
            # update how many times a token appeared in the document
            result[tokenid] = frequency

        # return tokenids, in ascending id order
        result = sorted(iteritems(result))
        return result

