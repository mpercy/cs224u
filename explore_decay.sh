#!/bin/bash -xe
for DECAY in 1 0.95 0.9 0.8 0.6 0.4 0.2; do
  ./explore_depth.sh $DECAY > log-explore-decay-$DECAY.log 2>&1 &
done
