import tensorflow as tf
import numpy as np


# Convert the model
saved_model_dir = 'C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door_mobilenetV2FPNLite/myFinalModeltfLiteSSDMobilenetFPN320/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.

with open('mobilenetV2FpnLiteUnquantized.tflite', 'wb') as f:
  f.write(tflite_model)

