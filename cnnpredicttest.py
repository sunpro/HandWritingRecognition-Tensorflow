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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import shutil
trained_dir = './cnncheckpoint/'
checkpoint_prefix = os.path.join(trained_dir, 'model')

import cv2

import tensorflow as tf

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')  
                        
x = tf.placeholder(tf.float32, [None, 784])
#y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
                        

keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 1, 32])
X_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#saver = tf.train.import_meta_graph("./cnncheckpoint/model-20000.meta")
saver = tf.train.Saver()
tf.global_variables_initializer().run()

ckpt = tf.train.get_checkpoint_state(trained_dir)
if ckpt and ckpt.model_checkpoint_path:
    #print('debug')
    #print(ckpt.model_checkpoint_path)
    saver.restore(sess, tf.train.latest_checkpoint('./cnncheckpoint/'))
    print(sess.run(W_conv1))
