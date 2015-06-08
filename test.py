#!/usr/bin/env python
##########################################################

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB



from util import topicSearch, chunkify

import logging
import numpy as np
import unittest
import random

logger = logging.getLogger("cs224u.test")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

class NaiveBayesBaseLine(unittest.TestCase):
    def testBaseLine(self):
        return # disable slow test for now

        news_train = load_mlcomp('20news-18828', 'train')
        news_test = load_mlcomp('20news-18828', 'test')
        vectorizer = TfidfVectorizer(encoding='latin1')
        X_train = vectorizer.fit_transform((open(f).read()
                                        for f in news_train.filenames))
        y_train = news_train.target
        X_test = vectorizer.transform((open(f).read() 
                                        for f in news_test.filenames))
        y_test = news_test.target

        clf = MultinomialNB().fit(X_train, y_train)
        pred = clf.predict(X_test)
        print(classification_report(y_test, pred,
                                    target_names=news_test.target_names))


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
    unittest.main()
