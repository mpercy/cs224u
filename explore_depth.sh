#!/bin/bash -e
DECAY=$1
if [ -z "$DECAY" ]; then
  echo "usage: $0 decay"
  exit 1
fi
set -x
for DEPTH in 0 1 2 3 4 5 6 7; do
  ./main.py --sample_size 0 --featurizer TopKLayerHierarchicalFeatureExtractor --depth "$DEPTH" --decay "$DECAY" glove.6B.300d.txt 20news-18828/ records-decay-$DECAY.txt
done
