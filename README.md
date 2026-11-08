# Robbie – Door Detection Training Pipeline

Training, evaluation and TensorFlow Lite conversion of the **door detectors** used in my
master's thesis *"Autonomous operation of a mobile robot using computer vision – machine learning"*
(Technische Hochschule Nürnberg Georg Simon Ohm, M.Sc. Electronic and Mechatronic Systems, Dec 2021,
industry partner: codemanufaktur GmbH).

The trained models run on an Android phone mounted on a small robot ("Robbie"). The phone detects a
door, estimates depth with MiDaS, and steers the robot via Bluetooth.
App code: [`robbie-door-navigation-android`](../robbie-door-navigation-android).

## What is in here

```
scripts/preprocessing/        dataset split (partition_dataset.py) and VOC XML -> TFRecord
workspace/
  training_eval_Robbie_door_mobilenetV2/          SSD MobileNet V2 320x320
  training_eval_Robbie_door_mobilenetV2FPNLite/   SSD MobileNet V2 FPN-Lite 320x320  <- used on the robot
  training_eval_Robbie_door_mobilenet640V2FPNLite/SSD MobileNet V2 FPN-Lite 640x640
  training_eval_Robbie_door_resnet50V1FPNLite/    SSD ResNet50 V1 FPN 640x640
    models/<name>/pipeline.config                 training config (TF2 Object Detection API)
    myFinalModeltfLite*/tfLiteConverter_*.py      SavedModel -> TFLite (float / int8)
    myFinalModeltfLite*/addMetadataTotfLite*.py   TFLite metadata (labels, normalisation)
    Evaluation_Results*.txt                       COCO metrics
    trainingAndTestStartDescription.txt           commands used for training / eval / TensorBoard
```

**Not included:** images, annotations, TFRecords, checkpoints and exported models (several GB, and the
images show non-public office space). The pipeline configs reference them by their original relative paths.

## Pipeline

1. **Label** images with [labelImg](https://github.com/tzutalin/labelImg) (Pascal VOC XML, single class `Door`).
2. **Split** 90/10 with `scripts/preprocessing/partition_dataset.py`.
3. **Convert** to TFRecord with `scripts/preprocessing/generate_tfrecord.py`.
4. **Fine-tune** COCO-pretrained SSD detectors with the TF2 Object Detection API
   (`model_main_tf2.py`, batch size 8, cosine-decay learning rate with warm-up).
5. **Export** with `export_tflite_graph_tf2.py` -> SavedModel -> `tfLiteConverter_*.py`
   (float32 and full-integer int8 with a representative dataset).
6. **Add metadata** (labels, input normalisation mean = std = 127.5) for the Android TFLite Task/Interpreter API.

## Datasets

| Dataset | Train | Val | Test |
|---|---|---|---|
| General door detection (doors + hard negatives: cabinets, picture frames, lockers, windows, wooden/glass/sliding doors) | 512 | 60 | 137 (codemanufaktur doors) |
| codemanufaktur door detection | 363 | 42 | 160 (doors from Erlangen Arcaden, TH Nürnberg, supermarkets) |

## Results (from the thesis)

| Model | Inference on phone, int8, CPU 4 threads | float32, CPU 4 threads |
|---|---|---|
| SSD MobileNet V2 FPN-Lite 320 | **50 ms** | 102 ms |
| SSD MobileNet V2 FPN-Lite 640 | 203 ms | 352 ms |
| SSD ResNet50 V1 FPN 640 | 1670 ms | 5116 ms |

COCO mAP@[.5:.95] on the *general* test set: 0.40–0.53 depending on the model.
The 320×320 FPN-Lite model was used on the robot: best trade-off between latency and accuracy.

## History

* Tag **`thesis-2021`** – the code exactly as used for the thesis.
* Later commits – fixes found when reviewing the code in 2026 (see `git log`), e.g. the int8
  calibration data was scaled to `[0, 1]` while the model expects `[-1, 1]`.

## Third-party code

* `model_main_tf2.py`, `exporter_main_v2.py`, `export_tflite_graph_tf2.py`, `create_pascal_tf_record.py`:
  [tensorflow/models](https://github.com/tensorflow/models) – Apache License 2.0, © The TensorFlow Authors.
* `partition_dataset.py`, `generate_tfrecord.py`: from the
  [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/).
* `addMetadataTotfLiteModel*.py`: adapted from the TensorFlow Lite metadata writer examples.
