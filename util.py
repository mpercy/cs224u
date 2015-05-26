from scipy.spatial import distance
import inspect
import logging
import os.path
import sys
from math import isnan
try:
   import cPickle as pickle
except:
   import pickle

from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk.tokenize import wordpunct_tokenize



sentenceEnds = ['...', '.', '.', '!', '?']

def sentenceSeg(doc):
    # new paragraph is meaningless here
    doc = doc.replace('\n', '').replace('\r', '')
    # split the doc with sentence ending marks
    initialRegions = [doc]
    for sentenceEnd in sentenceEnds:
        tmp = []
        for region in initialRegions:
            tmp += region.split(sentenceEnd)
        initialRegions = tmp
    return [x for x in initialRegions if x!='']


def cosine(x, y):
    rlt =  distance.cosine(x, y)
    if isnan(rlt):
        # TODO: what's the best
        return -2
    return rlt


def esa_model(line,
              dictionary = Dictionary.load_from_text('wiki_en_wordids.txt.bz2'),
              article_dict = pickle.load(open('wiki_en_bow.mm.metadata.cpickle', 'r')),
              tfidf = TfidfModel.load('wiki_en.tfidf_model'),
              similarity_index = Similarity.load('wiki_en_similarity.index', mmap='r')):
    doc = wordpunct_tokenize(utils.to_utf8(line).decode("utf8"))
    doc_bow = dictionary.doc2bow(doc)
    proc_doc = tfidf[doc_bow]
    sims = similarity_index[proc_doc]
    return sims

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
