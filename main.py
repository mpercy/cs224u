#!/usr/bin/env python
"""
Usage: %(program)s model_prefix data_dir

Run model.

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which must be in the current directory.

data_dir should be the base folder for the newsgroups data.

Example:
    %(program)s wiki_en 20news-18828
"""

from glove import GloveModel
from esa import ESAModel
from util import sentenceSeg, PriorityQueue, cosine, DataSet, function_name, \
                 MaxTopicFeatureExtractor, HierarchicalTopicFeatureExtractor, \
                 FlatFeatureExtractor, topicSearch, parseTree, getLayer
import argparse
import inspect
import json
import logging
import os.path
import sys
import numpy as np
import scipy.sparse

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

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

DEFAULT_MODEL = 'GloveModel'
DEFAULT_FEATURIZER = 'MaxTopicFeatureExtractor'
DEFAULT_NUM_REGIONS = 15


def main():
    testFullcoverage()

def testParseTree():
    tmp = pickle.load(open('d.pickle'))
    regs = tmp['regions']
    root = parseTree(regs, 32)
    layers = [root]
    while layers:
        log = ''
        nxtLayers = []
        for node in layers:
            s, e = node.region
            log += str(s)+','+str(e)+'; '
            for c in node.children:
                nxtLayers.append(c)
        print log
        layers = nxtLayers

def testFullcoverage():
    tmp = pickle.load(open('d.pickle'))
    regs = tmp['regions']
    root = parseTree(regs, 32)
    layers = getLayer(root, 4, fullCoverage=False)
    log = ''
    for node in layers:
        log += str(node.region[0]) + ',' + str(node.region[1])+ '; '
    print log


def parsingDoc():
    m = ESAModel('wiki_en-200000--20150531-035019')
    doc = open('/Users/Ted/Dropbox/2015_Spring/CS224U/cs224u/20news-18828/comp.os.ms-windows.misc/8514').read()

    initSeg, hypoLoc = topicSearch(doc, feature_extractor=m)
    print initSeg
    print hypoLoc


def evaluation(feature_extractor = None,
               clf = GaussianNB,
               model_prefix = None,
               data_dir = '20news-18828',
               result_record = None,
               record_fname = None):
    if result_record is None:
        raise Exception("Must pass result_record")
    if record_fname is None:
        raise Exception("Must pass record_fname")
    train = []
    trainY = []
    test = []
    testY = []

    # load data
    baseFolder = data_dir
    cats = listdir(baseFolder)
    for catIdx, cat in enumerate(cats):
        logger.info('Processing category %s (%d/%d)', cat, catIdx, len(cats))
        try:
            docs = listdir(os.path.join(baseFolder, cat))[:20]
        except:
            continue
        numDocs = len(docs)
        for docIdx, doc_filename in enumerate(docs):
            doc_filename = os.path.join(baseFolder, cat, doc_filename)
            logger.info('processing document %s (%d/%d)', doc_filename, docIdx, numDocs)
            doc = open(doc_filename).read()
            feature = feature_extractor.featurize(doc)
            logger.debug('doc %d feature extracted', docIdx)
            if docIdx < numDocs*0.9:
                train.append(feature)
                trainY.append(catIdx)
            else:
                test.append(feature)
                testY.append(catIdx)
            logger.debug('-----')

    # Convert to sparse format for compact storage and minimal memory usage.
    train = np.vstack(train)
    trainY = np.hstack(trainY)
    test = np.vstack(test)
    testY = np.hstack(testY)

    logger.info("Shape of training set: %s", train.shape)
    logger.info("Shape of test set: %s", test.shape)

    for clf_class in [GaussianNB, LogisticRegression, SVC]:
        classifier_name = function_name(clf_class)
        if classifier_name is None:
            raise Exception("Unable to get name of classifier class", clf_class)

        logger.info("Evaluating on classifier %s...", classifier_name)
        clf = clf_class()
        clf.fit(train, trainY)
        logger.info('training finished')

        # Make prediction.
        testPredY = clf.predict(test)

        # Print detailed report.
        print(classification_report(testY, testPredY, target_names = cats, digits = 5))

        # Save the important metrics.
        precision, recall, f1, support = \
            precision_recall_fscore_support(testY, testPredY, average='micro')
        result_record[classifier_name + "_precision"] = precision
        result_record[classifier_name + "_recall"] = recall
        result_record[classifier_name + "_f1"] = f1

    with open(record_fname, "a") as records_out:
        json.dump(result_record, records_out, sort_keys = True)
        records_out.write("\n")

def evaluate():
    # Define command-line args.
    parser = argparse.ArgumentParser(description='Evaluate topic classification approaches.',
                                     epilog=str(inspect.cleandoc(__doc__) % {'program': program}))
    parser.add_argument('--model', help=('Base feature model. Default: ' + DEFAULT_MODEL))
    parser.set_defaults(model=DEFAULT_MODEL)
    parser.add_argument('--featurizer',
                        help=('Higher level featurizer. Default: ' + DEFAULT_FEATURIZER))
    parser.set_defaults(featurizer=DEFAULT_FEATURIZER)
    parser.add_argument('--max_regions', type=int,
                        help=('Maximum regions to use. Default: ' + str(DEFAULT_NUM_REGIONS)))
    parser.set_defaults(max_regions=DEFAULT_NUM_REGIONS)
    parser.add_argument('--reverse', type=bool,
                        help=('Whether to reverse the hierarchical region iteration'))
    parser.set_defaults(reverse=True)
    parser.add_argument('model_prefix', help='Model prefix of passed to the model constructor')
    parser.add_argument('data_dir', help='Directory in which to find the 20-newsgroups data.')
    parser.add_argument('record_fname', help='Filename to append result records.')
    args = parser.parse_args()

    # load base feature model
    model_clazz = globals()[args.model]
    model = model_clazz(args.model_prefix)
    #model = ESAModel(args.model_prefix) # ESA is not working very well.
    #model = GloveModel(args.model_prefix)

    # load secondary feature extractor
    featurizer_clazz = globals()[args.featurizer]
    options = {'base_feature_extractor': model,
               'max_regions': args.max_regions,
               'reverse': args.reverse}
    featurizer = featurizer_clazz(options)
    #featurizer = MaxTopicFeatureExtractor(options)

    result_record = {}
    result_record['model_prefix'] = args.model_prefix
    result_record['model'] = args.model
    result_record['featurizer'] = args.featurizer
    result_record['max_regions'] = args.max_regions
    result_record['reverse'] = args.reverse

    evaluation(feature_extractor = featurizer,
               model_prefix = args.model_prefix,
               data_dir = args.data_dir,
               result_record = result_record,
               record_fname = args.record_fname)
if __name__ == '__main__':
    main()