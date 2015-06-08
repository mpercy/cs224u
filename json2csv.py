#!/usr/bin/env python
##########################################################
import json
import sys
import fileinput

allfields = set()
lines = []
for line in fileinput.input():
    lines.append(line)
    obj = json.loads(line)
    for k in obj:
        allfields.add(k)

fields = list(sorted(allfields))
numfields = len(fields)
sys.stdout.write("\t".join(fields))
sys.stdout.write("\n")

for line in lines:
    obj = json.loads(line)
    for i, field in enumerate(fields):
        sys.stdout.write("%s" % (obj[field] if field in obj else '',))
        if i < numfields - 1:
            sys.stdout.write("\t")
    sys.stdout.write("\n")
