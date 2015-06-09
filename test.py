#!/usr/bin/env python
##########################################################

from sklearn.datasets import load_mlcomp, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from models import LDAModel, LSAModel

from util import * # topicSearch, chunkify
from os import listdir

from distributedwordreps import ShallowNeuralNetwork

import logging
import numpy as np
import unittest
import random

logger = logging.getLogger("cs224u.test")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

def test():
    train = []
    trainY = []
    test = []
    testY = []

    # load data
    baseFolder = '20news-18828'
    opts = {'base_feature_extractor':LDAModel(), 'depth':5, 'fullLayer':True, 'decay':0.8}
    feature_extractor = TopKLayerHierarchicalFeatureExtractor(opts)
    cats = listdir(baseFolder)
    for catIdx, cat in enumerate(cats):
        logger.info('Processing category %s (%d/%d)', cat, catIdx, len(cats))
        try:
            docs = sorted(listdir(os.path.join(baseFolder, cat)), key = int)
            if sample_size is not None and sample_size != 0:
                docs = docs[:sample_size]
        except:
            continue
        numDocs = len(docs)
        print numDocs
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

    #for clf_class in [GaussianNB, MultinomialNB, LogisticRegression, SVC]:
    for clf_class in [LogisticRegression]:
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
            precision_recall_fscore_support(testY, testPredY, average='weighted')
        result_record[classifier_name + "_precision"] = precision
        result_record[classifier_name + "_recall"] = recall
        result_record[classifier_name + "_f1"] = f1
        #result_record[classifier_name + "_support"] = support

    with open(record_fname, "a") as records_out:
        json.dump(result_record, records_out, sort_keys = True)
        records_out.write("\n")


class NaiveBayesBaseLine(unittest.TestCase):
    def testBaseLine(self):
        return
        logger.info("Running 20NG NB baseline...")
        news_train = fetch_20newsgroups(subset='train')
        news_test = fetch_20newsgroups(subset='test')
        vectorizer = TfidfVectorizer(encoding='latin1')
        X_train = vectorizer.fit_transform(news_train.data)
        y_train = news_train.target
        X_test = vectorizer.transform(news_test.data)
        y_test = news_test.target

        clf = MultinomialNB().fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        print(classification_report(y_test, pred_test,
                                    target_names=news_test.target_names))
        logger.info("Done.")

class MockFeatureExtractor(object):
    def num_features(self):
        return 2

    def featurize(self, tokens):
        if random.choice([True, False]):
            return np.array([1, 2], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

class TopicSearchTest(unittest.TestCase):

    def testTopicSearch(self):
        doc = 'This is a doc. I have a clock. Hello Jim. What the duck? '
        feature_extractor = MockFeatureExtractor()
        segments, regions = topicSearch(doc, feature_extractor = feature_extractor)
        logging.info(segments)
        self.assertEqual(4, len(segments))
        for r in regions:
            logging.info(r)
        num_segments = len(segments)
        # one for each merge.
        expected_num_regions = num_segments + (num_segments - 1)
        self.assertEqual(expected_num_regions, len(regions))

    def testChunkify(self):
        s = "This is a long string with no gosh darn punctuation"
        chunks = chunkify(s, 8)
        self.assertGreater(len(chunks), 3, chunks)
        logger.info(chunks)

if __name__ == '__main__':
    test()
    # unittest.main()
