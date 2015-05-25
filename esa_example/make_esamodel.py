#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Copyright (C) 2015 Mike Percy, Bin Wang
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
If you have the `pattern` package installed, this script can use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern

Example: %(program)s enwiki-20150403-pages-articles.xml.bz2 wiki_en
"""

import argparse
import inspect
import logging
import os.path
import sys

from gensim.corpora import Dictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel
from gensim.similarities import Similarity

# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 100000

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # Define command-line args.
    parser = argparse.ArgumentParser(description='Build an ESA model from a Wikipedia dump.',
                                     epilog=str(__doc__ % {'program': program}))
    parser.add_argument('--lemmatize', dest='lemmatize', action='store_true', help='use the "pattern" lemmatizer')
    parser.set_defaults(lemmatize=False)
    parser.add_argument('--dict_size', type=int, help=('how many of the most frequent words to keep (after removing tokens that appear in more than 10%% of all documents). Defaults to ' + str(DEFAULT_DICT_SIZE)))
    parser.set_defaults(dict_size=DEFAULT_DICT_SIZE)
    parser.add_argument('input_file', help='filename of a bz2-compressed dump of Wikipedia articles, in XML format');
    parser.add_argument('output_prefix', help='the filename prefix for all output files');

    # Parse command-line args.
    args = parser.parse_args()
    lemmatize = args.lemmatize
    keep_words = args.dict_size
    input_file = args.input_file
    output_prefix = args.output_prefix

    logger.info(">>> Running %s" % ' '.join(sys.argv))
    logger.info("Lemmatize? %s" % lemmatize)
    logger.info("Dict size = %d" % keep_words)

    logger.info(">>> Loading Wikipedia corpus from %s ..." % input_file)
    wiki = WikiCorpus(input_file, lemmatize=lemmatize) # takes about 9h on a macbook pro, for 3.5m articles (june 2011)
    # only keep the most frequent words (out of total ~8.2m unique tokens)
    wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)

    # save dictionary and bag-of-words (term-document frequency matrix)
    logger.info(">>> Serializing bag-of-words representation of filtered wikipedia corpus ...")
    MmCorpus.serialize(output_prefix + '_bow.mm', wiki, progress_cnt=10000, metadata=True) # another ~9h
    wiki.dictionary.save_as_text(output_prefix + '_wordids.txt.bz2')
    del wiki

    # load back the id->word mapping directly from file
    # this seems to save more memory, compared to keeping the wiki.dictionary object from above
    dictionary = Dictionary.load_from_text(output_prefix + '_wordids.txt.bz2')

    # initialize corpus reader and word->id mapping
    mm = MmCorpus(output_prefix + '_bow.mm')

    # build tfidf, ~50min
    logger.info(">>> Building TF-IDF model ...")
    tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    tfidf.save(output_prefix + '.tfidf_model')

    """
    # save tfidf vectors in matrix market format
    # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
    logger.info(">>> Serializing TF-IDF vectors ...")
    MmCorpus.serialize(output_prefix + '_tfidf.mm', tfidf[mm], progress_cnt=10000)
    del tfidf
    tfidf_corpus = MmCorpus(output_prefix + '_tfidf.mm')
    """

    logger.info(">>> Generating similarity index ...")
    #similarity_index = Similarity(output_prefix + "_similarity_index", tfidf_corpus, len(dictionary))
    similarity_index = Similarity(output_prefix + "_similarity_index", tfidf[mm], len(dictionary))
    del tfidf

    logger.info(">>> Serializing similarity index ...")
    similarity_index.save(output_prefix + '_similarity.index')

    logger.info(">>> Finished running %s" % program)
