# Training and Evaluation

## Directory Structure for Training and Evaluation

```
# from mtl-ssl-detection/objct_detection

+configs
  +test
    -pipeline configuration files
+data
  -label_map file
  +mscoco
    -train TFRecord file
    -eval TFRecord file
  +voc
    -train TFRecord file
    -eval TFRecord file
+checkpoints
  +train
    +model
      -checkpoints file
  +eval
    +model
      -tensorflow summary file / detection results
```

## Running the Training Job

A local training job can be run with the following command:

```bash
from mtl-ssl-detection/objct_detection/scripts
bash run_train.sh ${MODEL_NAME}
```

where `${MODEL_NAME}` points to the model name (e.g. model11).


## Running the Evaluation Job

Evaluation is run as a separate job. The eval job will periodically poll the
train directory for new checkpoints and evaluate them on a test dataset. The
job can be run using the following command:

```bash
from mtl-ssl-detection/objct_detection/scripts
bash run_eval.sh ${MODEL_NAME}
```



## Running Tensorboard

Progress for training and eval jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}
```

where `${PATH_TO_MODEL_DIRECTORY}` points to the directory that contains the
train and eval directories. Please note it may take Tensorboard a couple minutes
to populate with data.
