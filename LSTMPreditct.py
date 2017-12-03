#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 滤掉乱七八糟的输出信息

import tensorflow as tf
import numpy as np
import cv2
# 定义checkpoin文件路径
checkpoint_dir = './cnncheckpoint1203/'
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

# Parameters
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
#y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
#    try:
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                           dtype=tf.float32)
#    except Exception: # Old TensorFlow version only returns outputs not states
#        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    

pred = BiRNN(x, weights, biases)
result_pred = tf.argmax(pred,1)
v1 = tf.get_variable("v1", shape=[3])
#创建Saver对象
saver = tf.train.Saver()
sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print(sess.run(weights))
    print(sess.run(v1))

for i in range(14):
    img = cv2.imread('./mnist6/mnist_' + str(i) + '.jpg',0)
    batch_xs1 = img.reshape((1,784))
    maxNum = max(batch_xs1[0])
    batch_xs = img /maxNum
    batch_xs = batch_xs.reshape((1,28,28))
    #print(batch_xs)
    numpre = sess.run(result_pred, feed_dict={x: batch_xs})
    print('predit number is : %d' %(numpre[0]))   
