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


def main():
    doc = open('unreasonable.txt').readline()
    print topicSearch(doc)

def topicSearch(doc, model=esa_model, similarity = cosine, initialPropose = sentenceSeg):
    # facilitating functions
    def getRegion(similarityArray, i, initSeg):
        if similarityArray[i] == 0:
            raise Exception("Not a region head...", "what's this for?")
        j = i
        while(j < len(similarityArray) and similarityArray[j]==0):
            j+=1
        return '.'.join(initSeg[i:j+1])
    def getPrevious(similarityArray, i):
        if i == 0:
             return None
        pre = i-1
        while(similarityArray[pre]==0):
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

    # loading model information
    dictionary = Dictionary.load_from_text('wiki_en_wordids.txt.bz2')
    article_dict = pickle.load(open('wiki_en_bow.mm.metadata.cpickle', 'r'))
    tfidf = TfidfModel.load('wiki_en.tfidf_model')
    similarity_index = Similarity.load('wiki_en_similarity.index', mmap='r')

    # initial proposal of regions
    initSeg = initialPropose(doc)
    # recording initial regions
    hypothesesLocations = [(i, i+1) for i in range(len(initSeg))]
    # similarity is recorded as an array the non-zero value is the start of a 
    # segment with its similarity to next
    similaritySet = [0 for _ in range(len(initSeg))]
    # to mark the last region as -1 
    similaritySet[-1] = -1
    print len(similaritySet), ' segments'

    # initialize similarity set.
    for i in range(len(similaritySet)-1):
        cur = model(initSeg[i], dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
	next = model(initSeg[i+1], dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
	similaritySet[i] = similarity(cur, next)
    print 'similarity initialized!'
    print similaritySet
    print 
    while(True):
        print similaritySet
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
            print 'pre idx:', preIdx
            pre = model(getRegion(similaritySet, preIdx, initSeg), dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)        
            similaritySet[preIdx] = similarity(pre, cur)
        nxtIdx = getNext(similaritySet, mostSimilar)
        if nxtIdx == None:
            similaritySet[mostSimilar] = -1
        else:
            nxt = model(getRegion(similaritySet, nxtIdx, initSeg), dictionary = dictionary, article_dict = article_dict, tfidf = tfidf, similarity_index = similarity_index)
            similaritySet[mostSimilar] = similarity(cur, nxt)
        print
        # add new region to hypotheses locations
        hypothesesLocations.append((mostSimilar, nxtIdx))

    return hypothesesLocations
        


if __name__ == "__main__":
    main()
