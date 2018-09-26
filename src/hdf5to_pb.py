"""
this script help you to convert hdf5 into .pb
"""

import argparse
import os

import tensorflow as tf

from keras.models import load_model
from keras import backend as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

"""
def generator_loss_yuv444(y_true, y_pred):
    return K.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0]) + 0.5*K.binary_crossentropy(y_true[:, :, :, 1], y_pred[:, :, :, 1]) + 0.5*K.binary_crossentropy(y_true[:, :, :, 2], y_pred[:, :, :, 2])
"""

def main(model_path, out_dir, num_out, prefix, name, readable):
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  K.set_learning_phase(0)
  # net_model = load_model(model_path, custom_objects={"generator_loss_yuv444" : generator_loss_yuv444})
  net_model = load_model(model_path)

  pred = [None]*num_out
  pred_node_names = [None]*num_out
  for i in range(num_out):
    pred_node_names[i] = prefix + '_' + str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
  print('Output nodes names are:', pred_node_names)

  sess = K.get_session()
  # Write the graph in human readable
  if readable:
    f = 'graph_reference.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), out_dir, f, as_text=True)
    print('Saved the graph definition in ascii at :', out_dir + '/' + f)
  
  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
  graph_io.write_graph(constant_graph, out_dir, name, as_text=False)
  print('Saved the constant graph at: ', out_dir + '/' + name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', '-m', required=True, help='REQUIRED: the hdf5 Keras model you wish to convert to .pb')
  parser.add_argument('--numout', type=int, required=True, help='REQUIRED: the number of outputs in the model')
  parser.add_argument('--outdir', '-o', default='./', help='the directory to place the output files')
  parser.add_argument('--prefix', default='biscotti', help='The prefix for the output aliasing')
  parser.add_argument('--name', default='output_graph.pb', help='The name of the resulting output graph')
  parser.add_argument('--readable', default=True, type=bool, help='Write the graph in human readable to outdir/')
  args = parser.parse_args()

  main(args.model, args.outdir, args.numout, args.prefix, args.name, args.readable)