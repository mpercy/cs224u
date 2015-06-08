#!/bin/bash -xe

for model_data in glove.6B.100d.txt glove.6B.300d.txt; do
  for featurizer in MaxTopicFeatureExtractor HierarchicalTopicFeatureExtractor FlatFeatureExtractor; do
    for max_regions in 1 5 15 30 50; do
      ./main.py --featurizer $featurizer --max_regions $max_regions $model_data 20news-18828/ records.txt
    done
  done
done
