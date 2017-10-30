# -*- coding: UTF-8 -*- 
import input_data

import tensorflow as tf 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 创建模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)


y_ = tf.placeholder(tf.float32, [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 训练模型
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver(tf.global_variables())

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

save_path = saver.save(sess, "model/model.ckpt", global_step=1000)
# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('=============')
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


