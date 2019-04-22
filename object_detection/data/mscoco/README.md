## MS COCO dataset

dataset homepage: http://cocodataset.org/


Download and unzip MS COCO images
``` bash
# from  mtl-ssl-detection/object_detection/data/mscoco/
mkdir images
cd images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

```

Download and unzip MS COCO annotations

``` bash
# from  mtl-ssl-detection/object_detection/data/mscoco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip annotations_trainval2017.zip
unzip image_info_test2017.zip
```


Install [COCO API](https://github.com/cocodataset/cocoapi)
``` bash
# from  mtl-ssl-detection/object_detection/data/mscoco/
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make install
```

Generating the MS COCO TFRecord files
``` bash
# from  mtl-ssl-detection/object_detection/create_records/
python create_mscoco_tf_record.py
```