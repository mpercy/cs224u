#!/usr/bin/env python
"""
Usage: %(program)s model_prefix

Validate that all documents in the reverse index match all the documents in the
forward index. This test can take a long time in a large index.

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which should be in the current
directory.

Example:
    %(program)s wiki_en
"""

import heapq
import inspect
import logging
import numpy as np
import os.path
import scipy
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

    logger.info("Loading document name map...")
    article_dict = utils.unpickle(model_prefix + '_bow.mm.metadata.cpickle')

    logger.info("Loading tf-idf model...")
    tfidf = TfidfModel.load(model_prefix + '.tfidf_model')

    logger.info("Loading similarity index...")
    similarity_index = Similarity.load(model_prefix + '_similarity.index', mmap='r')
    similarity_index.use_reverse_index = True

    logger.info("Finished loading model files.")

    mismatches = 0
    for doc_idx in range(0, len(similarity_index)):
        logger.info("Checking doc: %d %s" % (doc_idx, article_dict[doc_idx]))
        rev_doc = scipy.sparse.dok_matrix((1, len(dictionary)), dtype=np.float64)
        fwd_doc = similarity_index.vector_by_id(doc_idx)
        for feature_id, val in enumerate(fwd_doc.toarray().flatten()):
            if val == 0: continue
            feat_rev_docs = similarity_index.docs_by_feature_id(feature_id).toarray().flatten()
            rev_doc[0, feature_id] = feat_rev_docs[doc_idx]
        rev_doc = rev_doc.tocsr()

        if (fwd_doc - rev_doc).nnz > 0:
            mismatches += 1
            logger.info("> MISMATCH!")

            if not logger.isEnabledFor(logging.DEBUG):
                continue

            logger.debug("===========================")
            logger.debug("FORWARD:")
            for feature_id, val in enumerate(fwd_doc.toarray().flatten()):
                if val == 0: continue
                logger.debug("%d (%s): %f" % (feature_id, dictionary[feature_id], val))
            logger.debug("===========================")
            logger.debug("REVERSE:")
            for feature_id, val in enumerate(rev_doc.toarray().flatten()):
                if val == 0: continue
                logger.debug("%d (%s): %f" % (feature_id, dictionary[feature_id], val))
            logger.debug("===========================")

    if mismatches > 0:
        logger.info("Number of mismatches: %d" % (mismatches,))
        sys.exit(1)

    logger.info("ALL GOOD")
    sys.exit(0)
