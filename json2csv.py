#!/usr/bin/env python
##########################################################
import json
import sys

if len(sys.argv) < 2:
    print "Usage: " + sys.argv[0] + " infile"
    sys.exit(1)

fname = sys.argv[1]

allfields = set()
with open(fname, "rb") as infile:
    for line in infile:
        obj = json.loads(line)
        for k in obj:
            allfields.add(k)

fields = list(sorted(allfields))
numfields = len(fields)
sys.stdout.write("\t".join(fields))
sys.stdout.write("\n")
with open(fname, "rb") as infile:
    for line in infile:
        obj = json.loads(line)
        for i, field in enumerate(fields):
            sys.stdout.write("%s" % (obj[field] if field in obj else '',))
            if i < numfields - 1:
                sys.stdout.write("\t")
        sys.stdout.write("\n")
