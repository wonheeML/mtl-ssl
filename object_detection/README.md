# Multi-task Self-supervised Object Detection

This is an implementation of CVPR 2019: "Multi-task Self-supervised Object Detection via Recycling of Bounding Box Annotations"
This is a novel object detection approach that takes advantage of both multi-task learning (MTL) and self-supervised learning (SSL).
We propose a set of auxiliary tasks that help improve the accuracy of object detection.

The code is modified from [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).



## Table of contents

* <a href='g3doc/preparation.md'>
   Preparation</a><br>

* <a href='g3doc/exporting_models.md'>
   Exporting a trained model for inference</a><br>

* <a href='notebooks/object_detection_tutorial.ipynb'>
   Quick Start: Jupyter notebook for off-the-shelf inference</a><br>

* <a href='g3doc/train_and_eval.md'>
   Training and Evaluation</a><br>

## Evaluation of models

| Model name (baseline/ours)  | Detector | Backbone | Training | Eval | Baseline | Ours |
| ------------ |  :--------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| [model11](https://drive.google.com/open?id=1kgfRctafWG-IhJrlo7GyXqSrhPmFnnR9) / [model12](https://drive.google.com/file/d/1FA2hx4VQQYxeZFs6S3wecynknb2cewvS/view?usp=sharing) | Faster R-CNN | ResNet101 | VOC 07 trainval | VOC 07 test | 77.0 | 78.7 |
| [model21](https://drive.google.com/file/d/1_1SNcmHa5ubU1HlCt3DbuCxLdqTR3N26/view?usp=sharing) / [model22](https://drive.google.com/file/d/1IEVI_QuEoaDVop9-T3XBUNzpzOCYcD37/view?usp=sharing) | Faster R-CNN | ResNet101           | COCO 2017 train    | COCO 2017 val | 32.7 | 34.6 |
| [model31](https://drive.google.com/file/d/1hhiSU3IqneGu7ASVvRj8WR7IBkQ2j7uR/view?usp=sharing) / [model32](https://drive.google.com/file/d/1wXhSRmu3y2f1GvKCF-xj34IS589aGcA4/view?usp=sharing) | Faster R-CNN | ResNet101           | VOC 07+12 trainval | VOC 07 test   | 81.7 | 83.7 |
| [model41](https://drive.google.com/file/d/1mCxkbCAnyMZfZtHZ6RnOyQvzOeXtdas8/view?usp=sharing) / [model42](https://drive.google.com/file/d/1o77OTTqBV0OKrVE5xbHg5EEqoGuqpd7N/view?usp=sharing) | R-FCN        | ResNet101           | VOC 07 trainval    | VOC 07 test   | 73.5 | 74.7 |
| [model51](https://drive.google.com/file/d/1GDGINcl1O5y0k4TWdMjehcTzvUV5eMNa/view?usp=sharing) / [model52](https://drive.google.com/file/d/1mZFpiaZrWmvjenWfAzVM2bFZUsk22CSn/view?usp=sharing) | Faster R-CNN | MobileNet           | VOC 07 trainval    | VOC 07 test   | 61.2 | 63.8 |
| [model61](https://drive.google.com/file/d/1ipT2ozga0aCH4h5EvGCmt2qA_VSrTkBL/view?usp=sharing) / [model62](https://drive.google.com/file/d/1Nf4Ugpsgib-_19vTzwURlXPxXoQ49tts/view?usp=sharing) | Faster R-CNN | Inception ResNet v2 | VOC 07 trainval    | VOC 07 test   | 80.7 | 81.8 |
| [model71](https://drive.google.com/file/d/15gjbSbE5yL_sdUJErL0zS5tQIminEfG9/view?usp=sharing) / [model72](https://drive.google.com/file/d/15s9SNcgznhWf9pZve2yqd--fzC0kbG0S/view?usp=sharing) | R-FCN        | ResNet101           | VOC 07+12 trainval | VOC 07 test   | 78.6 | 80.6 |
| [model81](https://drive.google.com/file/d/1FApCJ0qWnmVxqVXhKRYk8hvB6W6eMUzd/view?usp=sharing) / [model82](https://drive.google.com/file/d/1XKAaDnOIxw7oEwTnmeJiSygGyvyWymXz/view?usp=sharing) | Faster R-CNN | MobileNet           | VOC 07+12 trainval | VOC 07 test   | 68.6 | 70.8 |
| [model91](https://drive.google.com/file/d/1yuzT_xdx0iqPIlY8CWFcOArE2AuXsIIq/view?usp=sharing) / [model92](https://drive.google.com/file/d/1KdbiEWu7wq0qlUDl62gGNvlfbEnsQFwx/view?usp=sharing) | Faster R-CNN | Inception ResNet v2 | VOC 07+12 trainval | VOC 07 test   | 84.3 | 86.0 |



