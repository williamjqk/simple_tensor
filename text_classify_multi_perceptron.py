import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])
    op = tf.add(x,y)
    result = sess.run(fetches=op)
    print(result)

# %%
n_hidden_1 = 10
n_hidden_2 = 5
n_input = total_words
n_classes = 3

def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)
    
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition

weights