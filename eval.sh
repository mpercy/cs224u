#!/bin/bash
DATA_DIR=$(basename "$1") # strip leading / trailing slashes
if [ ! -d "$DATA_DIR" ]; then
  echo "Usage: $0 data_dir"
  exit 1
fi

BASE=$(dirname $(readlink -f $0))

set -o pipefail # pass any errors through back to the shell
set -x

cd $DATA_DIR
$BASE/main.py $DATA_DIR $BASE/20news-18828 2>&1 | tee run_log_$(date '+%Y%m%d-%H%M%S').log
echo "Python retcode: $?"
