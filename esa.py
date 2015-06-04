#!/usr/bin/env python

import heapq
import logging
import sys

from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk.tokenize import wordpunct_tokenize

if __name__ == '__main__':
    sys.exit(0)

logger = logging.getLogger("ESA")

class ESAModel(object):
    """
    Encapsulates an ESA model.
    Provide a model prefix, and the class takes care of loading everything else.
    Sharded files are mmapped and read in lazily, so the first time accessing a
    particular shard will display some slowdown until the caches get warmed up.
    """

    def __init__(self, model_prefix=None, num_best = None):
        self.model_prefix = model_prefix
        self.num_best = num_best
        if self.model_prefix is None:
            raise ValueError("model_prefix must be specified")

        logger.info("ESA: Loading word dictionary...")
        self.dictionary = Dictionary.load_from_text(model_prefix + '_wordids.txt.bz2')

        logger.info("ESA: Loading document name map...")
        self.article_dict = utils.unpickle(model_prefix + '_bow.mm.metadata.cpickle')

        logger.info("ESA: Loading TF-IDF model...")
        self.tfidf = TfidfModel.load(model_prefix + '.tfidf_model')

        logger.info("ESA: Loading similarity index...")
        self.similarity_index = Similarity.load(model_prefix + '_similarity.index', mmap='r')

        #logger.info("ESA: Preloading reverse indexes...")
        #self.similarity_index.preload_reverse_index()

        logger.info("ESA: Finished loading model files.")

    def num_documents(self):
        """ Returns number of documents in the index. """
        return len(self.similarity_index)

    def num_terms(self):
        """ Returns the number of words / features / terms used in the vocabulary. """
        return len(self.dictionary)

    def get_similarity(self, input_str, num_best=None, use_reverse_index=True):
        """
        Returns similar documents by cosine similarity based on TF-IDF score.
        If num_best is left as None, returns a numpy.array with a score for
        every document in the corpus. Otherwise, it returns the top-K scored
        items as a list of (doc_idx, score) tuples.
        If use_reverse_index is set to False, the forward index is used (and
        the full corpus is queried). This is only a good idea when the number
        of terms in the input string is big, such as the text of a long article.
        For short documents, using the reverse index is usually much faster.
        """

        logger.debug("input string: %s", input_str)

        # Tokenize the input string.
        input_str = utils.to_utf8(input_str, errors='replace').decode("utf8")
        doc = wordpunct_tokenize(input_str)
        doc = [w.lower() for w in doc]

        # Convert from tokens to word ids from the model dictionary.
        doc_bow = self.dictionary.doc2bow(doc)

        # Get TF-IDF score for the document words (this does not update the TF-IDF model itself).
        doc_tfidf = self.tfidf[doc_bow]

        # Calculate similarity scores.
        self.similarity_index.use_reverse_index = use_reverse_index
        similar_docs = self.similarity_index[doc_tfidf]

        # Fall back to self.num_best if it wasn't specified here.
        if num_best is None:
            num_best = self.num_best
        if num_best is None:
            return similar_docs

        # Return top-k if requested.
        return heapq.nlargest(num_best, enumerate(similar_docs), key=lambda item: item[1])
