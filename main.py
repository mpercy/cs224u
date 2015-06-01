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

from esa import ESAModel
from util import sentenceSeg, PriorityQueue, cosine
import inspect
import logging
import os.path
import sys
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

from gensim import utils
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

    evaluation(model_prefix = model_prefix, data_dir = data_dir)

def convertToFeature(seg, regs, model = None):
    feature = [0.0 for _ in range(model.num_documents())]
    # cnt = 0
    for reg in regs:
        # print '\t', cnt
        # cnt += 1
        doc = ' '.join(seg[reg[0]:reg[1]])
        s = model.get_similarity(doc)
        feature = np.amax([s, feature], axis=0)
        # print feature.shape, s.shape
    return feature

def topicSearch(doc, model=None, similarity = cosine, initialPropose = sentenceSeg):
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
    # recording initial regions
    hypothesesLocations = [(i, i+1) for i in range(len(initSeg))]
    # similarity is recorded as an array the non-zero value is the start of a 
    # segment with its similarity to next
    similaritySet = [0 for _ in range(len(initSeg))]
    # to mark the last region as -1 
    similaritySet[-1] = -1
    # print len(similaritySet), ' segments'

    # initialize similarity set.
    for i in range(len(similaritySet)-1):
        cur = model.get_similarity(initSeg[i])
        # print len(cur), 'topics'
        # exit(1)
    next = model.get_similarity(initSeg[i+1])
    similaritySet[i] = similarity(cur, next)
    # print 'similarity initialized!'
    # print similaritySet
    # print 
    while(True):
        # print similaritySet
        # get the most similar
        mostSimilar = np.argmax(similaritySet)
        if similaritySet[mostSimilar] == 0:
            break

        # merge region
        similaritySet[getNext(similaritySet, mostSimilar)] = 0

        # set the similarity score properly
        cur = model.get_similarity(getRegion(similaritySet, mostSimilar, initSeg))
        preIdx = getPrevious(similaritySet, mostSimilar)
        if preIdx != None:
            # print 'pre idx:', preIdx
            pre = model.get_similarity(getRegion(similaritySet, preIdx, initSeg))
            similaritySet[preIdx] = similarity(pre, cur)
        nxtIdx = getNext(similaritySet, mostSimilar)
        if nxtIdx == None:
            similaritySet[mostSimilar] = -1
        else:
            nxt = model.get_similarity(getRegion(similaritySet, nxtIdx, initSeg))
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
    prediction = clf.predict(test)
    logger.info('trained')
    totalCnt = len(test)
    correctCnt = 0
    for idx in range(totalCnt):
        if prediction[idx] == testY[idx]:
            correctCnt += 1
    return (1.0*correctCnt)/totalCnt

def evaluation(clf = NaiveBayes, model_prefix = None, data_dir = '20news-18828'):
    train = []
    trainY = []
    test = []
    testY = []

    # load model
    model = ESAModel(model_prefix)

    # load data
    baseFolder = data_dir
    cats = listdir(baseFolder)
    for catIdx, cat in enumerate(cats):
        logger.info('processing category %s (%d/%d)', cat, catIdx, len(cats))
        try:
            docs = listdir(os.path.join(baseFolder, cat))[:20]
        except:
            continue
        numDocs = len(docs)
        for docIdx, doc_filename in enumerate(docs):
            cat_doc_filename = os.path.join(cat, doc_filename)
            logger.info('processing document %s (%d/%d)', cat_doc_filename, docIdx, numDocs)
            doc = open(os.path.join(baseFolder, cat_doc_filename)).read()
            seg, regs = topicSearch(doc, model = model)
            logger.info('doc %d segmented', docIdx)
            feature = convertToFeature(seg, regs, model = model)
            logger.info('doc %d feature extracted', docIdx)
            if docIdx < numDocs*0.9:
                train.append(feature)
                trainY.append(catIdx)
            else:
                test.append(feature)
                testY.append(catIdx)
            logger.info('-----')
    clf(train, trainY, test, testY)

if __name__ == "__main__":
    main()
