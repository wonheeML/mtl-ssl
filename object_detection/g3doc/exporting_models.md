# Exporting a trained model for inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from tensorflow/models/object_detection:

``` bash
# from mtl-ssl-detection/object_detection
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${CHECKPOINT_FILE_PATH} \
    --output_directory ${OUTPUT_DIR_PATH}
```
(e.g.)
``` bash
# from mtl-ssl-detection/object_detection
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./configs/test/model11.config \
    --trained_checkpoint_prefix ./checkpoints/train/model11/model.ckpt \
    --output_directory ./checkpoints/fronzen/model11
```

Afterwards, you should see a graph named output_inference_graph.pb.
