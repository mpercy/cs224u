#!/usr/bin/env python
"""
Usage: %(program)s model_prefix data_dir

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which must be in the current directory.

data_dir should be the base folder for the newsgroups data.

Example:
    %(program)s wiki_en 20news-18828
"""

from glove import GloveModel
from esa import ESAModel, ClusteredESAModel
from models import LDAModel, LSAModel
from util import sentenceSeg, PriorityQueue, cosine, DataSet, function_name, \
                 MaxTopicFeatureExtractor, HierarchicalTopicFeatureExtractor, \
                 FlatFeatureExtractor, TopKLayerHierarchicalFeatureExtractor, \
                 topicSearch
#from distributedwordreps import ShallowNeuralNetwork
import argparse
import inspect
import json
import logging
import os.path
import sys
import time
import numpy as np
import scipy.sparse

from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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

DEFAULT_MODEL = 'LDAModel' #'GloveModel'
DEFAULT_FEATURIZER = 'MaxTopicFeatureExtractor'
DEFAULT_NUM_REGIONS = 15
DEFAULT_SAMPLE_SIZE = 20

def runTopicSearchAndSave(dirpath, model, catIdx, numCats, sample_size):
    logger.info('Processing category %s (%d/%d)', dirpath, catIdx, numCats)
    try:
        filtered_docs = []
        for d in listdir(dirpath):
            if not d.endswith(".pickle"):
                filtered_docs.append(d)
        docs = sorted(filtered_docs, key = int)
        if sample_size is not None and sample_size != 0:
            docs = docs[:sample_size]
    except:
        logger.warning('Got some weird exception')
        return
    numDocs = len(docs)
    for docIdx, doc_filename in enumerate(docs):
        doc_filename = os.path.join(dirpath, doc_filename)
        pickle_filename = "%s.%s.segmented.pickle" % (doc_filename, model.__class__.__name__)
        logger.info('Segmenting document %s (%d/%d) and storing as %s', \
                    doc_filename, docIdx, numDocs, pickle_filename)
        doc = open(doc_filename).read()
        segments, regions = topicSearch(doc, feature_extractor = model)
        gensim.utils.pickle([segments, regions], pickle_filename)

def segment(model = None,
               clf = GaussianNB,
               model_prefix = None,
               data_dir = '20news-18828',
               result_record = None,
               record_fname = None,
               sample_size = None):

    # load data
    baseFolder = data_dir
    cats = sorted(listdir(baseFolder))
    numCats = len(cats)
    Parallel(n_jobs=-2)(delayed(runTopicSearchAndSave) \
            (os.path.join(baseFolder, cat), model, catIdx, numCats, sample_size) \
            for catIdx, cat in enumerate(cats))

    logger.info("Done.")

if __name__ == "__main__":

    # Define command-line args.
    parser = argparse.ArgumentParser(description='Run parallel topic segmentation job.',
                                     epilog=str(__doc__ % {'program': program}))
    parser.add_argument('--model', help=('Base feature model. Default: ' + DEFAULT_MODEL))
    parser.set_defaults(model=DEFAULT_MODEL)

    parser.add_argument('--sample_size', type=int,
                        help=('How much to sample the dataset. Set to 0 to disable sampling. Default: ' + str(DEFAULT_SAMPLE_SIZE)))
    parser.set_defaults(sample_size=DEFAULT_SAMPLE_SIZE)

    parser.add_argument('model_prefix', help='Model prefix of passed to the model constructor')
    parser.add_argument('data_dir', help='Directory in which to find the 20-newsgroups data.')
    args = parser.parse_args()

    # load base feature model
    model_clazz = globals()[args.model]
    model = model_clazz(args.model_prefix)
    #model = ESAModel(args.model_prefix) # ESA is not working very well.
    #model = GloveModel(args.model_prefix)

    segment(model = model,
            model_prefix = args.model_prefix,
            data_dir = args.data_dir,
            sample_size = args.sample_size)
