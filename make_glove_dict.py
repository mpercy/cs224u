#!/usr/bin/env python
###########################################################
from gensim.corpora import Dictionary
from util import SimpleDict

from os.path import basename
import logging
import sys

import numpy as np

program = sys.argv[0]
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s glove_file" % (sys.argv[0],))
        sys.exit(1)
    glove_file = sys.argv[1]

    d = SimpleDict()
    with open(glove_file, 'r') as file:
        for lineno, line in enumerate(file):
            if lineno % 10000 == 0:
                logger.info("Processing line %d", lineno)
            token, values_txt = line.rstrip().split(' ', 1)
            idx = len(d)
            d.token2id[token] = idx
            d.id2token.append(token)
            d.vectors.append(np.fromstring(values_txt, sep=' ', dtype=np.float64))
    d.finalize()
    outfile = "%s.pickle" % (basename(glove_file),)
    d.save(outfile)
