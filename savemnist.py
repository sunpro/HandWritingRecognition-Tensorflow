from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets(r'D:\tmp\MNIST_data', one_hot=True)

batch = mnist.test.next_batch(50)

#print(batch.shape)
print(batch[0].shape)
print(batch[1].shape)
print(batch[0][1].shape)

testbatch = mnist.test.next_batch(50)
imgall = np.vstack((batch[0],testbatch[0]))