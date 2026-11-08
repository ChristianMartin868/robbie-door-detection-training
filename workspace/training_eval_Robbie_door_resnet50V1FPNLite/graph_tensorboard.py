import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.compat.v1.Session() as sess:
  model_filename ='C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door/exported-models/my_model/saved_model/saved_model.pb'
  with gfile.FastGFile(model_filename, 'rb') as f:

    data = compat.as_bytes(f.read())
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)

    if 1 != len(sm.meta_graphs):
      print('More than one graph found. Not sure which to write')
      sys.exit(1)

    g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
LOGDIR='C:/repos/Tensorflow2_ObjectDetection/workspace/training_eval_Robbie_door/logdir'
train_writer = tf.contrib.summary.create_file_writer(LOGDIR)
#train_writer = tf.compat.v1.summary.FileWriter(LOGDIR, graph=sess.graph)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()