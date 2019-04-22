## PASCAL VOC dataset

dataset homepage: http://host.robots.ox.ac.uk/pascal/VOC/


Download and unzip PASCAL VOC dataset
``` bash
# from  mtl-ssl-detection/object_detection/data/voc/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

Generating the PASCAL VOC TFRecord files
``` bash
# from  mtl-ssl-detection/object_detection/create_records/
bash create_all_pascal_tf_records.sh
```