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
from util import sentenceSeg, PriorityQueue, cosine, DataSet, \
                 MaxTopicFeatureExtractor, HierarchicalTopicFeatureExtractor
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

def evaluation(feature_extractor = None,
               clf = NaiveBayes,
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

    for clf in [NaiveBayes, logisticRegression, SVM]:
        classifier_name = funcname(clf)
        logger.info("Evaluating on classifier %s...", classifier_name)
        result = clf(train, trainY, test, testY)
        logger.info("Fraction correct: %f", result)
        logger.info("========================")
        result_record[classifier_name] = result

    with open(record_fname, "a") as records_out:
        json.dump(result_record, records_out, sort_keys = True)
        records_out.write("\n")

if __name__ == "__main__":

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
    featurizer = featurizer_clazz(base_feature_extractor = model)
    #featurizer = MaxTopicFeatureExtractor(base_feature_extractor = model)

    result_record = {}
    result_record['model_prefix'] = args.model_prefix
    result_record['model'] = args.model
    result_record['featurizer'] = args.featurizer
    result_record['max_regions'] = args.max_regions

    evaluation(feature_extractor = featurizer,
               model_prefix = args.model_prefix,
               data_dir = args.data_dir,
               result_record = result_record,
               record_fname = args.record_fname)
