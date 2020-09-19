import math
import numpy as numpy
import h5py
import matplotlib.pyplot as pyplot
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict



def one_hot_matrix(labels, C):

    
    # Create a tf.constant equal to C (depth), name it 'C'
    C = tf.constant(C, name="C")
    
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    
    return one_hot

def ones(shape):
    
    # Create "ones" tensor using tf.ones(...).
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    

    return ones

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape=(n_x, None),name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None),name="Y")
    
    return X, Y

def initialize_parameters():

    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    

    Z1 = tf.add(tf.matmul(W1,X), b1)                                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                                          # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,Z1), b2)                                            # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                                          # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,Z2), b3)                                            # Z3 = np.dot(W3, A2) + b3

    
    return Z3


def compute_cost(Z3, Y):

    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    
    return cost