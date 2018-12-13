"""
保存复杂模型产生的softtarget
"""
import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)  
    input_checkpoint = checkpoint.model_checkpoint_path  
    # We precise the file fullname of our freezed graph  
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])  
    output_graph = absolute_model_folder + "/frozen_model_conv2d.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./checkpoint", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="y_conv", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph("MNIST_conv_model", args.output_node_names)



'''

#- - - -  - - - - - - -   - - - - - - - - -  --  - -- -    - -
import os, argparse  
import tensorflow as tf  
from tensorflow.python.framework import graph_util  
  
dir = os.path.dirname(os.path.realpath(__file__))  
 
def freeze_graph(model_folder):  
    # We retrieve our checkpoint fullpath  
    checkpoint = tf.train.get_checkpoint_state(model_folder)  
    input_checkpoint = checkpoint.model_checkpoint_path  
    # We precise the file fullname of our freezed graph  
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])  
    output_graph = absolute_model_folder + "/frozen_model_conv2d.pb"  
  
    # Before exporting our graph, we need to precise what is our output node  
    # this variables is plural, because you can have multiple output nodes  
    #freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点  
    #输出结点可以看我们模型的定义  
    #只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃  
    #所以,output_node_names必须根据不同的网络进行修改  
    output_node_names = "Accuracy/predictions"  
  
    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated  
    clear_devices = True  
      
    # We import the meta graph and retrive a Saver  
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)  
  
    # We retrieve the protobuf graph definition  
    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
  
    #We start a session and restore the graph weights  
    #这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen  
    #相当于将参数已经固化在了图当中   
    with tf.Session() as sess:  
        saver.restore(sess, input_checkpoint)  
  
        # We use a built-in TF helper to export variables to constant  
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names.split(",") # We split on comma for convenience  
        )   
  
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(output_graph, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
        print("%d ops in the final graph." % len(output_graph_def.node))  
  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_folder", type=str, help="Model folder to export")  
    args = parser.parse_args()  
  
    #freeze_graph("results/") #args.model_folder=
    freeze_graph("mnist_conv_model")
'''

