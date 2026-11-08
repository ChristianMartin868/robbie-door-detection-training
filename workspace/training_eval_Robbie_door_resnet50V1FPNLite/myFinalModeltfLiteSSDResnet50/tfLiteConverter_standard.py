import tensorflow as tf
import numpy as np


"""
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]
"""
#ZU Testzwecken
def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 320, 320, 3)
    yield [data.astype(np.float32)]


# Convert the model
saved_model_dir = 'C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_resnet50V1FPNLite/myFinalModeltfLiteSSDResnet50/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.

with open('resnet50.tflite', 'wb') as f:
  f.write(tflite_model)

