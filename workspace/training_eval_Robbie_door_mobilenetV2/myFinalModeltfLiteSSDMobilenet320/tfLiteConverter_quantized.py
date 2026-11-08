import tensorflow as tf
import numpy as np
import os
from PIL import Image
import sys



dataset_dir = "C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_mobilenetV2/images/train"
IMAGE_SIZE = 300


def representative_dataset(num_samples=200, seed=42):
  """Calibration data for full-integer quantization.

  Fixes compared to the thesis version (tag thesis-2021):
  * Input range: the detector is fed inputs normalised to [-1, 1]
    (mean = std = 127.5, see addMetadataTotfLiteModel*.py). The old code
    calibrated with [0, 1]. Because inference_input_type is uint8, the
    input quantisation parameters are derived from this calibration data,
    so the int8 model effectively mapped raw pixels to [0, 1] instead of
    [-1, 1] - a train/inference preprocessing mismatch.
    Verify: compare float32 vs int8 detections on the same test images.
  * Sampling: `next(iter(dataset_list))` created a new shuffled iterator
    on every loop pass, i.e. 200 random draws *with* replacement. Now the
    file list is shuffled once (seeded) and each image is used at most once.
  """
  files = tf.data.Dataset.list_files(dataset_dir + "/*.jpg", shuffle=True, seed=seed)
  for path in files.take(num_samples):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    yield [tf.expand_dims(image, 0)]

#ZU Testzwecken
"""
def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 320, 320, 3)
    yield [data.astype(np.float32)]
"""

# Convert the model
saved_model_dir = 'C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_mobilenetV2/myFinalModeltfLiteSSDMobilenet320/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model.

with open('mobilenetV2Quantized.tflite', 'wb') as f:
  f.write(tflite_model)

