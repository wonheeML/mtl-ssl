#!/bin/sh
export LD_PRELOAD="/usr/lib/libtcmalloc.so"

#DATA_DIR=$1
#OUTPUT_DIR=$2

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2007 \
#    --set=train \
#    --output_path=../data/voc/voc2007_train.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2007 \
#    --set=val \
#    --output_path=../data/voc/voc2007_val.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

python create_pascal_tf_record.py \
    --data_dir=../data/voc/VOCdevkit \
    --year=VOC2007 \
    --set=trainval \
    --output_path=../data/voc/voc2007_trainval.record \
    --label_map_path=../data/pascal_label_map.pbtxt

python create_pascal_tf_record.py \
    --data_dir=../data/voc/VOCdevkit \
    --year=VOC2007 \
    --set=test \
    --output_path=../data/voc/voc2007_test.record \
    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2012 \
#    --set=train \
#    --output_path=../data/voc/voc2012_train.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2012 \
#    --set=val \
#    --output_path=../data/voc/voc2012_val.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2012 \
#    --set=trainval \
#    --output_path=../data/voc/voc2012_trainval.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=merged \
#    --set=train \
#    --output_path=../data/voc/voc0712_train.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=merged \
#    --set=val \
#    --output_path=../data/voc/voc0712_val.record \
#    --label_map_path=../data/pascal_label_map.pbtxt

python create_pascal_tf_record.py \
    --data_dir=../data/voc/VOCdevkit \
    --year=merged \
    --set=trainval \
    --output_path=../data/voc/voc0712_trainval.record \
    --label_map_path=../data/pascal_label_map.pbtxt

#python create_pascal_tf_record.py \
#    --data_dir=../data/voc/VOCdevkit \
#    --year=VOC2012 \
#    --set=test \
#    --output_path=../data/voc/voc2012_test.record \
#    --label_map_path=../data/pascal_label_map.pbtxt