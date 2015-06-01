#!/usr/bin/env python
"""
Usage: %(program)s model_prefix

Make a reverse index from a forward index.

Example:
    %(program)s wiki_en
"""

import inspect
import logging
import os.path
import sys
import time

from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk.tokenize import wordpunct_tokenize

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # check and process input arguments
    if len(sys.argv) < 2:
        print(inspect.cleandoc(__doc__) % locals())
        sys.exit(1)
    model_prefix = sys.argv[1]

    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("Loading word dictionary...")
    dictionary = Dictionary.load_from_text(model_prefix + '_wordids.txt.bz2')
    logger.debug(dictionary)
    logger.info("Dictionary len: %d" % (len(dictionary),))

    logger.info("Loading document name map...")
    article_dict = utils.unpickle(model_prefix + '_bow.mm.metadata.cpickle')

    logger.info("Loading tf-idf model...")
    tfidf = TfidfModel.load(model_prefix + '.tfidf_model')

    logger.info("Loading similarity index...")
    similarity_index = Similarity.load(model_prefix + '_similarity.index', mmap='r')
    logger.info("Similarity num features: %d" % (similarity_index.num_features,))

    logger.info("Finished loading model files.")

    logger.info("Rebuilding reverse index...")
    similarity_index.rebuild_reverse_index(ri_shardsize=1024)
    similarity_index.save(model_prefix + '_similarity.index')
    logger.info("Finished rebuilding reverse index.")
