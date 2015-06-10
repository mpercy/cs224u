#!/bin/bash
##################################################
# Use to perform topic search on 20ng in parallel
##################################################
datadir=$1
shift
parallel=$1
shift

ROOT=$(readlink -f $(dirname $0))

if [ ! -d "$datadir" ]; then
    echo "Usage: $0 data_dir num_parallel [arguments for segment-topic-search.py]"
    exit 1
fi

for i in $(seq 1 $parallel); do
  cats[$i]=''
done

i=1
for d in $datadir/*; do
    d=$(basename $d)
    if [[ $i -gt $parallel ]]; then
        i=1
    fi
    if [ "${cats[$i]}" != "" ]; then
        cats[$i]="${cats[$i]},"
    fi
    cats[$i]="${cats[i]}$d"
    ((i++))
done

set -xe
for i in $(seq 1 $parallel); do
  $ROOT/segment-topic-search.py --cats "${cats[$i]}" $* &
done

wait
