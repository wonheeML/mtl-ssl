#!/bin/sh
export LD_PRELOAD="/usr/lib/libtcmalloc.so"

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -e|--eval_dir)
    EVAL_DIR="$2"
    shift # past argument
    ;;
    -c|--config_path)
    CONFIG_PATH="$2"
    shift # past argument
    ;;
    -p|--checkpoint_dir)
    CHECKPOINT_DIR="$2"
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

python ../eval.py \
    --logtostderr \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --pipeline_config_path=${CONFIG_PATH} \
    --eval_dir=${EVAL_DIR} \
    --gpu_fraction=0.0
