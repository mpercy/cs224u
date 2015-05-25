#!/usr/bin/env python
"""
Usage: %(program)s input_file model_prefix

input_file should be in the format of one document per line, with tokens
separated by spaces.

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which should be in the current
directory.

Example:
    %(program)s test.txt wiki_en
"""

import inspect
import logging
import os.path
import sys

try:
   import cPickle as pickle
except:
   import pickle

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
    if len(sys.argv) < 3:
        print(inspect.cleandoc(__doc__) % locals())
        sys.exit(1)
    input_file, output_prefix = sys.argv[1:3]

    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("Loading word dictionary...")
    dictionary = Dictionary.load_from_text(output_prefix + '_wordids.txt.bz2')
    logger.debug(dictionary)

    logger.info("Loading document name map...")
    article_dict = pickle.load(open(output_prefix + '_bow.mm.metadata.cpickle', 'r'))

    logger.info("Loading tf-idf model...")
    tfidf = TfidfModel.load(output_prefix + '.tfidf_model')

    logger.info("Loading similarity index...")
    similarity_index = Similarity.load(output_prefix + '_similarity.index', mmap='r')

    logger.info("Finished loading model files.")

    logger.info("Processing input documents...")

    try:
        infile = open(input_file, 'r')
    except IOError:
        print 'cannot open', input_file

    for docnum, line in enumerate(infile):
        logger.info("Processing document #%d..." % (docnum,))

        # Perform a simple tokenization of the document.
        doc = wordpunct_tokenize(line)
        logger.debug(doc)

        # Convert from tokens to word ids from the model dictionary.
        doc_bow = dictionary.doc2bow(doc)
        logger.debug(doc_bow)

        # Get TF-IDF score for the document words (this does not update the TF-IDF model itself).
        proc_doc = tfidf[doc_bow]
        logger.debug(proc_doc)

        # Calculate document cosine similarity against the Wikipedia concept corpus using
        # the document's TF-IDF word scores calculated in the previous step.
        similarity_index.num_best = 50 # Only include the top 50 concept matches.
        sims = similarity_index[proc_doc]
        logger.debug(sims)

        # Print the similarity scores in descending order.
        sims = sorted(sims, key=lambda item: -item[1])
        for doc_idx, similarity in sims:
            pageid, title = article_dict[doc_idx]
            print "Similarity %f: %s [doc-index %d, wiki-page-id %s]" % (similarity, title, doc_idx, pageid)

