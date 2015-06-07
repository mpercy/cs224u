#!/usr/bin/env python
##########################################################

from util import topicSearch

import logging
import numpy as np
import unittest
import random

logger = logging.getLogger("cs224u.test")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

class MockFeatureExtractor(object):
    def num_features(self):
        return 2

    def featurize(self, tokens):
        if random.choice([True, False]):
            return np.array([1, 2], dtype=np.float64)
        return np.zeros(2, dtype=np.float64)

class TopicSearchTest(unittest.TestCase):

    def testTopicSearch(self):
        doc = 'This is a doc. I have a clock. Hello Jim. What the duck? ;;;'
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

if __name__ == '__main__':
    unittest.main()
