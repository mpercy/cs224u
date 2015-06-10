#!/usr/bin/env python
######################################
import re
import sys
import gensim
from util import parseTree, getLayer


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: %s doc_filename num_layers" % sys.argv[0]
        sys.exit(1)

    doc_filename = sys.argv[1]
    num_layers = int(sys.argv[2])

    doc = gensim.utils.unpickle(doc_filename)
    segments = doc[0]
    regions = doc[1]

    root = parseTree(regions, len(segments))
    for layer in range(num_layers):
        layer_regs = [t.region for t in getLayer(root, layer)]
        print
        print "Layer: %d" % layer
        for i, reg in enumerate(layer_regs):
            print
            print "  Region %d: %s" % (i, reg)
            reg_str = " ".join(segments[ reg[0] : reg[1] ])
            reg_str = reg_str.replace("\n", " ")
            reg_str = re.sub(r'\s+', ' ', reg_str)
            print "    %s" % reg_str
