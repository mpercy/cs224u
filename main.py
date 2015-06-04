#!/usr/bin/env python
"""
Usage: %(program)s model_prefix data_dir

Run model.

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which must be in the current directory.

data_dir should be the base folder for the newsgroups data.

Example:
    %(program)s wiki_en
"""

from glove import GloveModel
from esa import ESAModel
from util import sentenceSeg, PriorityQueue, cosine, DataSet
import inspect
import logging
import os.path
import sys
import numpy as np
import scipy.sparse

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
   import cPickle as pickle
except:
   import pickle

import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk.tokenize import wordpunct_tokenize
from os import listdir

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

def main():
    # check and process input arguments
    if len(sys.argv) < 3:
        print(inspect.cleandoc(__doc__) % locals())
        sys.exit(1)
    model_prefix, data_dir = sys.argv[1:3]

    # load model
    #model = ESAModel(model_prefix)
    model = GloveModel(model_prefix + ".pickle")

    evaluation(model = model, model_prefix = model_prefix, data_dir = data_dir)

def convertToFeature(seg, regs, model = None):
    feature = np.zeros(shape=model.num_features(), dtype=np.float64)
    # cnt = 0
    for reg in regs:
        # print '\t', cnt
        # cnt += 1
        doc = ' '.join(seg[reg[0]:reg[1]])
        s = model.featurize(doc)
        feature = np.amax([s, feature], axis=0)
        # print feature.shape, s.shape
    return feature

def topicSearch(doc, model=None, similarity = cosine, initialPropose = sentenceSeg):
    logger.info("performing topic search...")
    """ attempt to merge adjacent sentences based on their model similarity """

    # facilitating functions
    def getRegion(similarityArray, i, initSeg):
        if similarityArray[i] == 0 and i!=0:
            raise Exception("Not a region head...", "what's this for?")
        j = i
        while(j < len(similarityArray) and similarityArray[j]==0):
            j+=1
        return '.'.join(initSeg[i:j+1])

    def getPrevious(similarityArray, i):
        if i == 0:
             return None
        pre = i-1
        while(similarityArray[pre]==0 and pre != 0):
               pre -= 1
        return pre

    def getNext(similarityArray, i):
        l = len(similarityArray)
        next = i+1
        while(next<l and similarityArray[next]==0):
            next += 1
        if next >= l:
            return None
        return next



    # initial proposal of regions
    initSeg = initialPropose(doc)
    logging.info("Created %d initial segments", len(initSeg))

    # recording initial regions
    hypothesesLocations = [(i, i+1) for i in range(len(initSeg))]

    # Similarity set is a list of similarities between a segment and its next segment.
    similaritySet = np.zeros(shape=(len(initSeg) - 1), dtype=np.float64)

    # Initialize similarities.
    for i in range(len(similaritySet)):
        cur = model.featurize(initSeg[i])
        next = model.featurize(initSeg[i+1])
        similaritySet[i] = similarity(cur, next)
    logger.info('Similarity initialized!')

    while True:
        logger.info("Segment similarities: %s", similaritySet)
        # get the most similar
        mostSimilar = np.argmax(similaritySet)
        if similaritySet[mostSimilar] == 0:
            break

        # merge region
        similaritySet[getNext(similaritySet, mostSimilar)] = 0

        # set the similarity score properly
        cur = model.featurize(getRegion(similaritySet, mostSimilar, initSeg))
        preIdx = getPrevious(similaritySet, mostSimilar)
        if preIdx != None:
            # print 'pre idx:', preIdx
            pre = model.featurize(getRegion(similaritySet, preIdx, initSeg))
            similaritySet[preIdx] = similarity(pre, cur)
        nxtIdx = getNext(similaritySet, mostSimilar)
        if nxtIdx == None:
            similaritySet[mostSimilar] = -1
        else:
            nxt = model.featurize(getRegion(similaritySet, nxtIdx, initSeg))
            similaritySet[mostSimilar] = similarity(cur, nxt)
        # print
        # add new region to hypotheses locations
        hypothesesLocations.append((mostSimilar, nxtIdx))

    return (initSeg, hypothesesLocations)

def SVM(train, trainY, test, testY):
    clf = SVC()
    clf.fit(train, trainY)
    prediction = clf.predict(test)
    logger.info('training finished')
    totalCnt = len(test)
    correctCnt = 0
    for idx in range(totalCnt):
        if prediction[idx] == testY[idx]:
            correctCnt += 1
    return (1.0*correctCnt)/totalCnt

def logisticRegression(train, trainY, test, testY):
    clf = LogisticRegression()
    clf.fit(train, trainY)
    prediction = clf.predict(test)

    totalCnt = len(test)
    correctCnt = 0
    for idx in range(totalCnt):
        if prediction[idx] == testY[idx]:
            correctCnt += 1
    return (1.0*correctCnt)/totalCnt

def NaiveBayes(train, trainY, test, testY):
    clf = GaussianNB()
    clf.fit(train, trainY)

    logger.info("Predicting...")
    prediction = clf.predict(test)
    logger.info('trained')
    totalCnt = len(test)
    correctCnt = 0
    for idx in range(totalCnt):
        if prediction[idx] == testY[idx]:
            correctCnt += 1
    return (1.0*correctCnt)/totalCnt

def funcname(f):
    for attr in inspect.getmembers(f):
        if attr[0] == '__name__':
            return attr[1]
    return None

def evaluation(model = None, clf = NaiveBayes, model_prefix = None, data_dir = '20news-18828'):
    train = []
    trainY = []
    test = []
    testY = []

    # load data
    baseFolder = data_dir
    cats = listdir(baseFolder)
    for catIdx, cat in enumerate(cats):
        logger.info('==========================================================')
        logger.info('Processing category %s (%d/%d)', cat, catIdx, len(cats))
        try:
            docs = listdir(os.path.join(baseFolder, cat))[:20]
        except:
            continue
        numDocs = len(docs)
        for docIdx, doc_filename in enumerate(docs):
            doc_filename = os.path.join(baseFolder, cat, doc_filename)
            logger.info('processing document %s (%d/%d)', doc_filename, docIdx, numDocs)
            logger.info('also things')
            doc = open(doc_filename).read()
            seg, regs = topicSearch(doc, model = model)
            logger.info('doc %d segmented', docIdx)
            feature = scipy.sparse.csr_matrix(convertToFeature(seg, regs, model = model))
            logger.info("Feature shape: %s, nnz: %d", feature.shape, feature.nnz)
            logger.info('doc %d feature extracted', docIdx)
            if docIdx < numDocs*0.9:
                train.append(feature)
                trainY.append(catIdx)
            else:
                test.append(feature)
                testY.append(catIdx)
            logger.info('-----')
    logger.info('==========================================================')

    # Convert to sparse format for compact storage and minimal memory usage.
    train = scipy.sparse.vstack(train, format='csr')
    trainY = scipy.sparse.vstack(trainY, format='csr')
    test = scipy.sparse.vstack(test, format='csr')
    testY = scipy.sparse.vstack(testY, format='csr')

    # Serialize to disk in an efficient, mmap-able format.
    dataset = DataSet(train, trainY, test, testY)
    filename = "dataset_" + model_prefix + ".pickle"
    logger.info("Saving dataset...")
    dataset.save(filename)
    # Free the memory for the existing data structures..
    del dataset, train, trainY, test, testY

    # Reload the dataset, mmapped.
    dataset = DataSet.load(filename, mmap='r')

    for clf in [NaiveBayes, logisticRegression, SVM]:
        logger.info("Evaluating on classifier %s...", funcname(clf))
        res = clf(dataset.train, dataset.trainY, dataset.test, dataset.testY)
        logger.info("Fraction correct: %f", res)
        logger.info("========================")

if __name__ == "__main__":
    main()
