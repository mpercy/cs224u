from scipy.spatial import distance
import sklearn.metrics.pairwise
import logging
import os.path
import re
import sys
from math import isnan
from six import iteritems

import itertools
import gensim
import numpy as np

sentenceEnds = ['...', '.', ';', '!', '?']
sentenceEndPattern = re.compile('|'.join([re.escape(tok) for tok in sentenceEnds]))

logger = logging.getLogger("cs224u.util")

def sentenceSeg(doc):
    # new paragraph is meaningless here
    doc = re.sub(r'\s+', ' ', doc)
    # split the doc with sentence ending marks
    initialRegions = re.split(sentenceEndPattern, doc)
    return [x.strip() for x in initialRegions if x != '']

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

    def num_features(self):
        return self.vectors.shape[1]

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

# Perform topic search and return a list of tuples containing progressively merged topics.
def topicSearch(doc, feature_extractor=None, similarity = cosine, initialPropose = sentenceSeg):
    logger.debug("performing topic search...")
    """ attempt to merge adjacent sentences based on their feature_extractor similarity """

    # Get a joined region of sentences starting at pairIndex and continuing as
    # long as the pair similarities are 0, which means either they are disjoint
    # or have been merged.
    def getRegion(pairSimilarities, pairIndex, segments):
        if pairSimilarities[pairIndex] == 0 and pairIndex != 0:
            raise Exception("Similarity of pair at index %d is 0: pair=('%s', '%s'), segments: %s" %
                            (pairIndex, segments[pairIndex], segments[pairIndex+1], segments))
        nextIndex = pairIndex
        while nextIndex < len(pairSimilarities) and pairSimilarities[nextIndex] == 0:
            nextIndex += 1
        return '. '.join(segments[pairIndex:nextIndex+1])

    # Returns the first index in pairSimilarities less than pairIndex in which
    # the pair similarity is nonzero, or None if it can't find one.
    def getPrevious(pairSimilarities, pairIndex):
        pairIndex -= 1
        while pairIndex >= 0 and pairSimilarities[pairIndex] == 0:
            pairIndex -= 1
        if pairIndex < 0:
            return None
        return pairIndex

    # Returns the next index in pairSimilarities after pairIndex in which the
    # pair similarity is nonzero, or None if it can't find one.
    def getNext(pairSimilarities, pairIndex):
        pairIndex += 1
        while pairIndex < len(pairSimilarities) and pairSimilarities[pairIndex] == 0:
            pairIndex += 1
        if pairIndex >= len(pairSimilarities):
            return None
        return pairIndex

    # initial proposal of regions
    initSeg = initialPropose(doc)
    logging.info("Created %d initial segments", len(initSeg))

    # recording initial regions
    hypothesesLocations = [(i, i+1) for i in range(len(initSeg))]

    # Similarity set is a list of similarities between a segment and its next segment.
    similaritySet = np.zeros(shape=(len(initSeg) - 1), dtype=np.float64)

    # Initialize similarities.
    for i in range(len(similaritySet)):
        curSegment = feature_extractor.featurize(initSeg[i])
        nextSegment = feature_extractor.featurize(initSeg[i+1])
        similaritySet[i] = similarity(curSegment, nextSegment)
    #logger.info('Similarity initialized!')

    while True:
        #logger.info("Segment similarities: %s", similaritySet)
        # get the most similar
        mostSimilarIndex = np.argmax(similaritySet)
        if similaritySet[mostSimilarIndex] == 0:
            break

        # Attempt to merge region.
        nextIndex = getNext(similaritySet, mostSimilarIndex)
        if nextIndex is not None:
            similaritySet[nextIndex] = 0

        # Recalculate similarity scores.
        curSegFeatures = feature_extractor.featurize(getRegion(similaritySet, mostSimilarIndex, initSeg))

        prevIndex = getPrevious(similaritySet, mostSimilarIndex)
        if prevIndex != None:
            # print 'pre idx:', prevIndex
            prevFeatures = feature_extractor.featurize(getRegion(similaritySet, prevIndex, initSeg))
            similaritySet[prevIndex] = similarity(prevFeatures, curSegFeatures)

        nextIndex = getNext(similaritySet, mostSimilarIndex)
        if nextIndex == None:
            similaritySet[mostSimilarIndex] = -1
        else:
            nextFeatures = feature_extractor.featurize(getRegion(similaritySet, nextIndex, initSeg))
            similaritySet[mostSimilarIndex] = similarity(curSegFeatures, nextFeatures)

        # add new region to hypotheses locations
        hypothesesLocations.append((mostSimilarIndex, nextIndex))

    return (initSeg, hypothesesLocations)

# Rename of convertToFeature() function.
# Performs an element-wise max on the extracted feature vectors.
def piecewiseMaxFeatures(tokens, regions, feature_extractor = None):
    feature = np.zeros(shape=feature_extractor.num_features(), dtype=np.float64)
    # cnt = 0
    for region_start, region_end in regions:
        # print '\t', cnt
        # cnt += 1
        doc = ' '.join(tokens[region_start:region_end])
        s = feature_extractor.featurize(doc)
        feature = np.amax([s, feature], axis=0)
        # print feature.shape, s.shape
    return feature

# Take the last 15 regions
def mergeHierarchicalSegments(tokens, regions, feature_extractor = None, max_regions = 15):
    features_per_region = feature_extractor.num_features()
    tot_num_features = features_per_region * max_regions
    doc_vec = np.zeros(shape=tot_num_features, dtype=np.float64)
    for i in range(len(regions)):
        if i >= max_regions:
            break
        # Iterate in reverse order.
        idx = len(regions) - 1 - i
        region_start, region_end = regions[idx]
        doc = ' '.join(tokens[region_start:region_end])
        region_vec = feature_extractor.featurize(doc)
        doc_offset = i * features_per_region
        doc_vec[doc_offset:doc_offset + features_per_region] = region_vec
    return doc_vec

class MaxTopicFeatureExtractor(object):
    def __init__(self, base_feature_extractor = None):
        if base_feature_extractor is None:
            raise Exception("model must be specified")
        self.feature_extractor = base_feature_extractor

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        tokens, regions = topicSearch(doc, feature_extractor = self.feature_extractor)
        feature = piecewiseMaxFeatures(tokens, regions, feature_extractor = self.feature_extractor)
        return feature

class HierarchicalTopicFeatureExtractor(object):
    def __init__(self, base_feature_extractor = None, max_regions = 15):
        if base_feature_extractor is None:
            raise Exception("model must be specified")
        self.feature_extractor = base_feature_extractor
        self.max_regions = max_regions

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        tokens, regions = topicSearch(doc, feature_extractor = self.feature_extractor)
        features = mergeHierarchicalSegments(tokens,
                                             regions,
                                             feature_extractor = self.feature_extractor,
                                             max_regions = self.max_regions)
        return features

class FlatFeatureExtractor(object):
    def __init__(self, base_feature_extractor = None):
        if base_feature_extractor is None:
            raise Exception("model must be specified")
        self.feature_extractor = base_feature_extractor

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        return self.feature_extractor.featurize(doc)

def function_name(f):
    return f.__name__
