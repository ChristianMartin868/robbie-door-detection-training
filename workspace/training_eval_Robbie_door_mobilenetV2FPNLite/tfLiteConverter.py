import tensorflow as tf

# Convert the model
saved_model_dir = 'C:/repos/TensorFlow2_last/workspace/training_demo/exported-models/my_model/saved_model'
#saved_model_dir = "C:/repos/TensorFlow2_last/workspace/training_demo/pre-trained-models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model.

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

