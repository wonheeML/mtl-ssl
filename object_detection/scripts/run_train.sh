#!/bin/sh
export LD_PRELOAD="/usr/lib/libtcmalloc.so"

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -t|--train_dir)
    TRAIN_DIR="$2"
    shift # past argument
    ;;
    -c|--config_path)
    CONFIG_PATH="$2"
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

python ../train.py \
    --logtostderr \
    --pipeline_config_path=${CONFIG_PATH} \
    --train_dir=${TRAIN_DIR} \
    --train_tag= \
    --num_clones=1
