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

import heapq
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
    if len(sys.argv) < 3:
        print(inspect.cleandoc(__doc__) % locals())
        sys.exit(1)
    input_file, output_prefix = sys.argv[1:3]

    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("Loading word dictionary...")
    dictionary = Dictionary.load_from_text(output_prefix + '_wordids.txt.bz2')
    logger.debug(dictionary)

    logger.info("Loading document name map...")
    article_dict = utils.unpickle(output_prefix + '_bow.mm.metadata.cpickle')

    logger.info("Loading tf-idf model...")
    tfidf = TfidfModel.load(output_prefix + '.tfidf_model')

    logger.info("Loading similarity index...")
    similarity_index = Similarity.load(output_prefix + '_similarity.index', mmap='r')
    similarity_index.use_reverse_index = True
    similarity_index.preload_reverse_index()

    logger.info("Finished loading model files.")

    logger.info("Processing input documents...")

    try:
        infile = open(input_file, 'r')
    except IOError:
        print('cannot open %s' % (input_file,))
        sys.exit(1)

    for docnum, line in enumerate(infile):
        line = line.rstrip()
        logger.info("Processing document #%d..." % (docnum,))

        # Perform a simple tokenization of the document.
        line = utils.to_utf8(line, errors='replace').decode("utf8")
        doc = wordpunct_tokenize(line)
        doc = [w.lower() for w in doc]
        logger.debug(doc)

        # Convert from tokens to word ids from the model dictionary.
        doc_bow = dictionary.doc2bow(doc)
        logger.debug(doc_bow)

        # Get TF-IDF score for the document words (this does not update the TF-IDF model itself).
        proc_doc = tfidf[doc_bow]
        logger.debug(proc_doc)

        # Calculate document cosine similarity against the Wikipedia concept corpus using
        # the document's TF-IDF word scores calculated in the previous step.
        NUM_BEST = 40

        saved_sims = []
        #for ri in [False, True]:
        for ri in [True]:
            similarity_index.use_reverse_index = ri
            for i in range(0, 3):
                logger.info("===============================================")
                logger.info("Performing query %d with reverse_indexes=%s" % (i, ri))
                start = time.time()
                sims = similarity_index[proc_doc]
                end = time.time()
                logger.info("Time elapsed: %s" % (end - start))
                logger.debug(sims)

            saved_sims.append(sims)

            # Print the similarity scores in descending order.
            print("LINE: %s" % (line,))
            sims = heapq.nlargest(NUM_BEST, enumerate(sims), key=lambda item: item[1])
            for doc_idx, similarity in sims:
                pageid, title = article_dict[doc_idx]
                print("Similarity %f: %s [doc-index %d, wiki-page-id %s]" %
                    (similarity, title, doc_idx, pageid))

