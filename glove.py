#!/usr/bin/env python
#####################################################################

from nltk.tokenize import wordpunct_tokenize
from util import SimpleDict

import gensim
import numpy as np

class GloveModel(object):
    def __init__(self, fname):
        self.fname = fname
        if self.fname is None:
            raise ValueError("fname must be specified")
        self.dict = SimpleDict.load(fname, mmap='r')

    def num_features(self):
        return self.dict.num_features()

    def featurize(self, input_str):
        input_str = gensim.utils.to_utf8(input_str, errors='replace').decode("utf8")
        doc = wordpunct_tokenize(input_str)
        doc = [w.lower() for w in doc]

        # Convert from tokens to word ids from the model dictionary.
        doc_bow = self.dict.doc2bow(doc)

        # Simply add up all the vectors and return.
        vec = np.zeros(shape=self.num_features(), dtype=np.float64())
        for tokenid, count in doc_bow:
            vec += count * self.dict.vectors[tokenid]
        return vec
