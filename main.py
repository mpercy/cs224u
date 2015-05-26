from util import sentenceSeg, PriorityQueue, esa_model, cosine
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

# loading model information
dictionary = Dictionary.load_from_text('wiki_en_wordids.txt.bz2')
article_dict = pickle.load(open('wiki_en_bow.mm.metadata.cpickle', 'r'))
tfidf = TfidfModel.load('wiki_en.tfidf_model')
similarity_index = Similarity.load('wiki_en_similarity.index', mmap='r')



def main():
    evaluation()

def convertToFeature(seg, regs, 
                     model = esa_model, 
                     # dictionary=dictionary, 
                     # article_dict=article_dict,
                     # tfidf = tfidf, 
                     # similarity_index = similarity_index, 
                     classNum = 3796181):
    feature = [0.0 for _ in range(classNum)]
    # cnt = 0
    for reg in regs:
        # print '\t', cnt
        # cnt += 1
        doc = ' '.join(seg[reg[0]:reg[1]])
        s = model(doc, dictionary, article_dict, tfidf, similarity_index)
        feature = np.amax([s, feature], axis=0)
        # print feature.shape, s.shape
    return feature

def topicSearch(doc, model=esa_model, similarity = cosine, initialPropose = sentenceSeg):
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
        cur = model(initSeg[i], dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
        # print len(cur), 'topics'
        # exit(1)
    next = model(initSeg[i+1], dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
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
        cur = model(getRegion(similaritySet, mostSimilar, initSeg), dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
        preIdx = getPrevious(similaritySet, mostSimilar)
        if preIdx != None:
            # print 'pre idx:', preIdx
            pre = model(getRegion(similaritySet, preIdx, initSeg), dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)        
            similaritySet[preIdx] = similarity(pre, cur)
        nxtIdx = getNext(similaritySet, mostSimilar)
        if nxtIdx == None:
            similaritySet[mostSimilar] = -1
        else:
            nxt = model(getRegion(similaritySet, nxtIdx, initSeg), dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
            similaritySet[mostSimilar] = similarity(cur, nxt)
        # print
        # add new region to hypotheses locations
        hypothesesLocations.append((mostSimilar, nxtIdx))

    return (initSeg, hypothesesLocations)
        

def SVM(train, trainY, test, testY):
    clf = SVC()
    clf.fit(train, trainY)
    prediction = clf.predict(test)
    print 'training finished'
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
    print 'trained'
    totalCnt = len(test)
    correctCnt = 0
    for idx in range(totalCnt):
        if prediction[idx] == testY[idx]:
            correctCnt += 1
    return (1.0*correctCnt)/totalCnt

def evaluation(tp_search = topicSearch, clf = NaiveBayes, topicSearch = topicSearch):
    train = []
    trainY = []
    test = []
    testY = []

    # load model
    dictionary = Dictionary.load_from_text('wiki_en_wordids.txt.bz2')
    article_dict = pickle.load(open('wiki_en_bow.mm.metadata.cpickle', 'r'))
    tfidf = TfidfModel.load('wiki_en.tfidf_model')
    similarity_index = Similarity.load('wiki_en_similarity.index', mmap='r')
    print 'model loaded'

    # load data
    baseFolder = '20news-18828'
    cats = listdir(baseFolder)
    for catIdx in range(len(cats)):
        print 'processing cat:', cats[catIdx]
        try:
            docs = listdir(baseFolder+'/'+cats[catIdx])[:20]
        except:
            continue
        docNum = len(docs)
        for i in range(docNum):
            print 'processing doc', i
            doc = open(baseFolder+'/'+cats[catIdx]+'/'+docs[i]).read()
            seg, regs = topicSearch(doc)
            print 'doc', i, 'segmented'
            feature = convertToFeature(seg, regs)
            print 'doc', i, 'feature extracted'
            if i < docNum*0.9:
                train.append(feature)
                trainY.append(catIdx)
            else:
                test.append(feature)
                testY.append(catIdx)
            print '-----'
            print
    clf(train, trainY, test, testY)

if __name__ == "__main__":
    main()
