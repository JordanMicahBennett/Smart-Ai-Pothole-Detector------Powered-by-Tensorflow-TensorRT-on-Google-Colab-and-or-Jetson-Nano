###################
#File created by Jordan Bennett
#Part B/3a. Load tensor RT fp16 graph
#This is not expected to work on windows.
#Expected to work on google colab or jetson nano like platform.
###################
import tensorflow.compat.v1 as tf

#jordan_declaration: input and output names taken from frozen graph generation process
#this is to avoid re-running frozen graph generation on jetson nano, instead okay to use generatred frozen graph file on storage.
input_names =  ['input_1', 'input_2']
output_names_ = ['lambda_2/Identity']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


#trt_graph = get_frozen_graph('pothole_model_tensor_rt_format/trt_pothole_graph.pb') #reads from colab directory
trt_graph = get_frozen_graph('trt_pothole_graph.pb') #reads from google drive

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))


# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names_[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

with tf.Session() as sess: #jordan_node added these two lines to resolve FailedPreconditionError, that happens in Part B/4 prediciton on runtime.
  tf_sess.run(tf.global_variables_initializer())

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)
