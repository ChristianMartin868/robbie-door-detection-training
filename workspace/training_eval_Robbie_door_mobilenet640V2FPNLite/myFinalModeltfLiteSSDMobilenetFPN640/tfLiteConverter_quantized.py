import tensorflow as tf
import numpy as np
import os
from PIL import Image
import sys



dataset_dir = "C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_mobilenet640V2FPNLite/images/train"
IMAGE_SIZE = 640


def representative_dataset():
  dataset_list = tf.data.Dataset.list_files(dataset_dir + "/*.jpg")
  for i in range(200):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

#ZU Testzwecken
"""
def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 320, 320, 3)
    yield [data.astype(np.float32)]
"""

# Convert the model
saved_model_dir = 'C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_mobilenet640V2FPNLite/myFinalModeltfLiteSSDMobilenetFPN640/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model.

with open('mobilenet640V2FpnLiteQuantized.tflite', 'wb') as f:
  f.write(tflite_model)

