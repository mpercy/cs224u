#!/usr/bin/env python
###################################################

import sys
import numpy as np

if len(sys.argv) < 2:
    print "Usage: %s assignments.npy" % sys.argv[0]
    sys.exit(1)

fname = sys.argv[1]

assignments = np.load(fname, mmap_mode='r')
num_clusters = 2000
cluster_counts = np.zeros(num_clusters)
for doc in assignments:
    cluster_counts[doc] += 1
for i, count in enumerate(cluster_counts):
    print "Cluster %d count: %d" % (i, count)
