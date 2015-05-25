#!/usr/bin/env python
"""
Usage: query_esa.py input_file model_prefix

input_file should be in the format of one document per line, with tokens
separated by spaces.

model_prefix should be something like "wiki_en" corresponding to the
filename prefix of the ESA model files, which should be in the current
directory.

Example:
    query_esa.py test.txt wiki_en
"""

import logging
import os.path
import sys

try:
   import cPickle as pickle
except:
   import pickle

from gensim.corpora import Dictionary, TextCorpus
from gensim.models import TfidfModel
from gensim.similarities import Similarity

class DocPerTextLineCorpus(TextCorpus):
    def __init__(self, fname, dictionary=None):
        if not dictionary:
            logger.fatal("no dictionary specified")

        try:
            self.infile = open(fname, 'r')
        except IOError:
            print 'cannot open', fname
        else:
            print fname, 'has', len(self.infile.readlines()), 'lines'

        self.metadata = False
        self.dictionary = dictionary

    def get_texts(self):
        for line in self.infile:
            yield line.strip().split()

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # check and process input arguments
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    input_file, output_prefix = sys.argv[1:3]

    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("Loading word dictionary...")
    dictionary = Dictionary.load_from_text(output_prefix + '_wordids.txt.bz2')
    logger.debug(dictionary)

    logger.info("Loading document name map...")
    article_dict = pickle.load(open(output_prefix + '_doc_index.pickle', 'r'))

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

        # Perform simple tokenization of the document by spaces.
        doc = line.strip().split()
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
        for doc_id, similarity in sims:
            doc_title = article_dict[doc_id]
            print "Similarity %f: %s [docid %d]" % (similarity, doc_title, doc_id)

