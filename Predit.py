'''
训练模型
'''
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
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'D:\desktop\TensorFlow\cnn\MNIST_DATA/', one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import shutil
import cv2
img = cv2.imread('0.jpg',0)
batch_xs = img.reshape((1,784))
maxNum = max(batch_xs[0])
batch_xs = batch_xs / maxNum
#print(batch_xs)
trained_dir = './checkpoint/'
checkpoint_prefix = os.path.join(trained_dir, 'model-1000')
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
sess = tf.InteractiveSession()
#保存模型
#saver = tf.train.import_meta_graph("./checkpoint/model-1000.meta")
saver = tf.train.Saver([W,b])
tf.global_variables_initializer().run()

sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(trained_dir)
if ckpt and ckpt.model_checkpoint_path:
    #print('debug')
    #print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    ww = sess.run(W)
    #print(max(ww.reshape((1,7840))[0]))
    #print(sess.run(b))
#batch_xs = mnist.train.next_batch(1)
#batch_xs = tf.reshape(img,[1,1,784,1])
#print(batch_xs[0])
#result = sess.run(y, feed_dict={x: batch_xs[0]})
result = tf.argmax(y, 1)

#y.run({x: batch_xs})
num = sess.run(result, feed_dict={x: batch_xs})
print('result' + str(num[0]))

