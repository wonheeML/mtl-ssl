# Preparation

## Installation
This has been tested on Ubuntu 16.04 python 2.7 environment.

Source code
``` bash
git clone https://github.com/wonheeML/mtl-ssl-detection.git
```

 
Requirements
``` bash
# from mtl-ssl-detection/
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.7.0-cp27-none-linux_x86_64.whl
pip install -r requirements.txt
```


Check the protoc version >= 3.5.1 [[protoc](http://google.github.io/proto-lens/installing-protoc.html)]
``` bash
protoc --version

# if < 3.5.1,
PROTOC_ZIP=protoc-3.5.1-linux-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
rm -f $PROTOC_ZIP
sudo chmod +rx /usr/local/bin/protoc

```


Protobuf Compilation
``` bash
# from from mtl-ssl-detection/
protoc object_detection/protos/*.proto --python_out=.
```


Add Libraries to PYTHONPATH
``` bash
# from from mtl-ssl-detection/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.


## Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
# from from mtl-ssl-detection/
python object_detection/builders/model_builder_test.py
```


# Dataset
* <a href='data/mscoco/README.md'>MS COCO</a><br>

* <a href='data/voc/README.md'>PASCAL VOC</a><br>


# Model Zoo
[tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/slim)
provide a collection of detection models pre-trained on the ImageNet.
The following models were tested.

``` bash
# from from mtl-ssl-detection/object_detection/checkpoints/detection_model_zoo/
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar -xf resnet_v1_101_2016_08_28.tar.gz
tar -xf inception_resnet_v2_2016_08_30.tar.gz 
mkdir mobilenet_v1_1.0_224
tar -xvf mobilenet_v1_1.0_224.tgz -C mobilenet_v1_1.0_224
```