"""
Usage: graph_freeze.py MODEL_DIR NODE_NAMES ...

Arguments:
    MODEL_DIR   path to tensorflow checkpoint
    NODE_NAMES  list of nodes to export from graph

Based on: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
"""
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import tensorflow as tf

from docopt import docopt


def freeze_graph(model_dir, node_names):
    '''
    Extracts the sub-graph defined by the nodes and converts
    all its variables into constants
    '''
    logger = logging.getLogger(__name__)

    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export directory: %s" % model_dir)

    if not node_names:
        logger.warning("You need to supply the name of a node")
        return

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # I don't know why, but even if training code is run in the docker container, the input
    # path is always the host absolute path.
    if not tf.gfile.Exists(input_checkpoint):
        input_checkpoint = os.path.join(model_dir, 'model')

    # We precise the file fullname of our freezed graph
    output_graph = os.path.join(os.path.dirname(model_dir), "frozen_model.pb")

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporaty fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(
            "%s.meta" % input_checkpoint, clear_devices=True)

        # Restore the weights
        saver.restore(sess, input_checkpoint)

        # use built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,                                   # session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes 
            node_names)                             # The nodes are used to select the subgraphs nodes

        # Serialize and dump the output subgraph to the filesystem
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        logger.info("%d ops in the source graph." % len(tf.get_default_graph().get_operations()))
        logger.info("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    args = docopt(__doc__, version='text')
    freeze_graph(args['MODEL_DIR'], args['NODE_NAMES'])
