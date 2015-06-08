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

class TopicTree:
    '''
        defines a tree structure for topics
        region is the starting and ending postion of current topic
        childeren are the topics of next level
    '''
    def __init__(self, region=None):
        self.region = region
        self.children = []

    def dump(self):
        log = str(self.region[0])+','+str(self.region[1])+': '
        for c in self.children:
            log += str(c.region[0]) + ',' + str(c.region[1]) + ';'

        print log


def parseTree(regs, length):
    '''
        this function parses the regs into a tree
        regs are the proposed regions, 
        length is the document length which is also the largest number in the regs
        this will return a 
    '''
    regions = dict()
    for start, end in regs:
        try:
            regions[start].append(end)
        except:
            regions[start] = [end]
    if length not in regions[0]:
        raise ValueError('Document is not fully covered. Topic search error!')
    root = TopicTree((0, length))
    regions[0].remove(length)

    def findChildren(node):
        s, e = node.region
        if e == s+1:
            return
        nxt = s
        while nxt < e:
            nxte = max(regions[nxt])
            tmp = TopicTree((nxt, nxte))
            node.children.append(tmp)
            regions[nxt].remove(nxte)
            nxt = nxte

        for child in node.children:
            findChildren(child)
    findChildren(root)
    return root

def getLayer(root, depth, fullCoverage = True):
    layer = [root]
    for i in range(depth):
        tmp = []
        for node in layer:
            if node.children:
                for c in node.children:
                    tmp.append(c)
            elif fullCoverage:
                tmp.append(node)
        layer = tmp
    return layer

def splitAfter(pattern, inputStr):
    ''' Split a string after each 'pattern'. Do not remove any characters. '''
    pieces = []
    start = 0
    inputLen = len(inputStr)
    splitPoints = re.finditer(pattern, inputStr)
    while start < inputLen:
        try:
            end = splitPoints.next().end()
        except StopIteration:
            end = inputLen
        pieces.append(inputStr[start:end])
        start = end
    return pieces

def sentenceSeg(doc):
    ''' Segment a document into sentences. '''
    # new paragraph is meaningless here
    doc = re.sub(r'\s+', ' ', doc)
    # split the doc with sentence ending marks
    initialSegments = splitAfter(sentenceEndPattern, doc)
    segments = []
    for s in initialSegments:
        stripped = s.strip()
        if stripped != '':
            segments.append(stripped)
    return segments

def cosine(x, y):
    rlt =  distance.cosine(x, y)
    if isnan(rlt):
        # TODO: what's the best
        return -2
    return rlt

def cosine_sparse(x, y):
    return sklearn.metrics.pairwise.cosine_similarity(x, y)

def function_name(f):
    return f.__name__

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
def topicSearch(doc, feature_extractor=None, similarity = cosine, splitter = sentenceSeg):
    logger.debug("performing topic search...")
    """ attempt to merge adjacent sentences based on their feature_extractor similarity """

    NEGATIVE_INFINITY = float("-inf")

    # Returns the first index in similarityWithNext less than pairIndex in which
    # the pair similarity is not -Inf, or 0 if there are none.
    def getPrevious(similarityWithNext, pairIndex):
        assert pairIndex < len(similarityWithNext), pairIndex
        pairIndex -= 1
        while pairIndex >= 0 and similarityWithNext[pairIndex] == NEGATIVE_INFINITY:
            pairIndex -= 1
        if pairIndex < 0:
            pairIndex = 0
        return pairIndex

    # Returns the next index in similarityWithNext after pairIndex in which the
    # pair similarity is not -Inf, or len(similarityWithNext) if it can't find one.
    def getNext(similarityWithNext, pairIndex):
        assert pairIndex >= 0, pairIndex
        pairIndex += 1
        while pairIndex < len(similarityWithNext) and similarityWithNext[pairIndex] == NEGATIVE_INFINITY:
            pairIndex += 1
        return pairIndex

    def joinSegments(segments, start, end):
        return ' '.join(segments[start:end])

    def mergeWithNext(similarityWithNext, pairIndex, segments):
        logger.debug("Merging index %d with next...", pairIndex)
        assert pairIndex >= 0 and pairIndex < len(similarityWithNext), pairIndex
        nextIndex = getNext(similarityWithNext, pairIndex)

        # Special-casing for when there is no next segment to merge.
        if nextIndex == len(similarityWithNext):
            if pairIndex == 0:
                # We special-case the situation where we are trying to merge
                # region 0 and there are no other regions to merge it with.
                # This will be our last merge.
                similarityWithNext[pairIndex] = NEGATIVE_INFINITY
                return None
            else:
                # Step backward to 'eat' this segment.
                return mergeWithNext(similarityWithNext, getPrevious(similarityWithNext, pairIndex), segments)

        # Consider nextIndex merged into pairIndex.
        similarityWithNext[nextIndex] = NEGATIVE_INFINITY

        # After destroying nextIndex, recalculate the new "next index" and the
        # features for our new merged region.
        nextIndex = getNext(similarityWithNext, pairIndex)
        curSegFeatures = feature_extractor.featurize(joinSegments(segments, pairIndex, nextIndex))

        # Calculate similarity to our new following neighbor.
        if nextIndex == len(similarityWithNext):
            # We use similarity of 0 to mark that no segments follow.
            similarityWithNext[pairIndex] = 0
        else:
            nextNextIndex = getNext(similarityWithNext, nextIndex)
            nextSegFeatures = feature_extractor.featurize(joinSegments(segments, nextIndex, nextNextIndex))
            similarityWithNext[pairIndex] = similarity(curSegFeatures, nextSegFeatures)

        # Recalculate similarity to our new previous neighbor.
        prevIndex = getPrevious(similarityWithNext, pairIndex)
        if prevIndex != pairIndex:
            # print 'pre idx:', prevIndex
            prevFeatures = feature_extractor.featurize(joinSegments(segments, prevIndex, pairIndex))
            similarityWithNext[prevIndex] = similarity(prevFeatures, curSegFeatures)

        return (pairIndex, nextIndex)

    # initial proposal of regions
    segments = splitter(doc)
    logging.info("Created %d initial segments", len(segments))

    # record initial regions
    regions = [(i, i+1) for i in range(len(segments))]

    # Similarity set is a list of similarities between a segment and its next segment.
    similarityWithNext = np.zeros(shape=len(segments), dtype=np.float64)

    # Initialize similarities.
    for i in range(len(similarityWithNext)):
        if i + 1 == len(segments):
            similarityWithNext[i] = 0
        else:
            curSegment = feature_extractor.featurize(segments[i])
            nextSegment = feature_extractor.featurize(segments[i+1])
            similarityWithNext[i] = similarity(curSegment, nextSegment)
    #logger.info('Similarity initialized!')

    while True:
        #logger.info("Segment similarities: %s", similarityWithNext)
        # Find and merge the most similar regions.
        mostSimilarIndex = np.argmax(similarityWithNext)
        if similarityWithNext[mostSimilarIndex] == NEGATIVE_INFINITY:
            # We have merged all of our regions.
            break

        # Merge the region.
        # Sometimes the merge can choose a different index to merge than the
        # one we asked for, so we need to record the one it returns.
        mergedRegion = mergeWithNext(similarityWithNext, mostSimilarIndex, segments)

        if mergedRegion is not None:
            # Add new region to hypotheses locations.
            regions.append(mergedRegion)

    return (segments, regions)

# Rename of convertToFeature() function.
# Performs an element-wise max on the extracted feature vectors.
def piecewiseMaxFeatures(segments, regions, feature_extractor = None):
    feature = np.zeros(shape=feature_extractor.num_features(), dtype=np.float64)
    # cnt = 0
    for region_start, region_end in regions:
        # print '\t', cnt
        # cnt += 1
        doc = ' '.join(segments[region_start:region_end])
        s = feature_extractor.featurize(doc)
        feature = np.amax([s, feature], axis=0)
        # print feature.shape, s.shape
    return feature

# Take the last 15 regions.
# This function takes a simple "unrolling" approach to hierarchical features.
def mergeHierarchicalSegments(segments, regions, feature_extractor = None, max_regions = 15, reverse = True):
    features_per_region = feature_extractor.num_features()
    tot_num_features = features_per_region * max_regions
    doc_vec = np.zeros(shape=tot_num_features, dtype=np.float64)
    for i in range(len(regions)):
        if i >= max_regions:
            break
        idx = i
        if reverse:
            # Iterate in reverse order.
            idx = len(regions) - 1 - i
        region_start, region_end = regions[idx]
        doc = ' '.join(segments[region_start:region_end])
        region_vec = feature_extractor.featurize(doc)
        doc_offset = i * features_per_region
        doc_vec[doc_offset:doc_offset + features_per_region] = region_vec
    return doc_vec

# returns the top k layer similarities
# the importance of each layer decays with its depth which may be fine tuned
# TODO: testing
def topKHierarchicalSegments(tokens, regions, feature_extractor = None, layers = 1, fullLayer = True, decay = 0.6):
    root = parseTree(regions, len(tokens)) #TODO: check whether it should be len(tokens) or len(tokens) - 1
    features = []
    alph = 1.
    for i in range(layers + 1):
        # print 'Layer', i,'Regions:', [t.region for t in getLayer(root, i, fullLayer)]
        regs = [t.region for t in getLayer(root, i, fullLayer)]
        features = np.hstack([features, piecewiseMaxFeatures(tokens, regs, feature_extractor)*alph])
        alph *= decay
    return features


class MaxTopicFeatureExtractor(object):
    def __init__(self, opts):
        if opts['base_feature_extractor'] is None:
            raise Exception("model must be specified")
        self.feature_extractor = opts['base_feature_extractor']

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        segments, regions = topicSearch(doc, feature_extractor = self.feature_extractor)
        feature = piecewiseMaxFeatures(segments, regions, feature_extractor = self.feature_extractor)
        return feature

class HierarchicalTopicFeatureExtractor(object):
    def __init__(self, opts):
        if opts['base_feature_extractor'] is None:
            raise Exception("model must be specified")
        self.feature_extractor = opts['base_feature_extractor']
        self.max_regions = opts['max_regions'] if 'max_regions' in opts else 15
        self.reverse = opts['reverse'] if 'reverse' in opts else False

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        segments, regions = topicSearch(doc, feature_extractor = self.feature_extractor)
        features = mergeHierarchicalSegments(segments,
                                             regions,
                                             feature_extractor = self.feature_extractor,
                                             max_regions = self.max_regions,
                                             reverse = self.reverse)
        return features

# TODO: testing
class TopKLayerHierarchicalFeatureExtractor(object):
    def __init__(self, opts):
        if opts['base_feature_extractor'] is None:
            raise Exception("model must be specified")
        self.feature_extractor = opts['base_feature_extractor']
        self.depth = opts['depth'] if 'depth' in opts else 7
        self.fullLayer = opts['fullLayer'] if 'fullLayer' in opts else True
        self.decay = opts['decay'] if 'decay' in opts else 0.35

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        tokens, regions = topicSearch(doc, feature_extractor = self.feature_extractor)
        features = topKHierarchicalSegments( tokens,
                                             regions,
                                             feature_extractor = self.feature_extractor,
                                             layers = self.depth,
                                             fullLayer = self.fullLayer,
                                             decay = self.decay)
        return features



class FlatFeatureExtractor(object):
    def __init__(self, opts):
        if opts['base_feature_extractor'] is None:
            raise Exception("model must be specified")
        self.feature_extractor = opts['base_feature_extractor']

    def num_features(self):
        return self.feature_extractor.num_features()

    def featurize(self, doc):
        doc = " ".join(sentenceSeg(doc))
        return self.feature_extractor.featurize(doc)
