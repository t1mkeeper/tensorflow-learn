# -*- coding: utf-8 -*-

from genIDCard import *
import tensorflow as tf
import cv2

obj = gen_id_card()

#图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = obj.max_size
CHAR_SET_LEN = obj.len

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
 
	# 4 conv layer
	w_c1 = tf.Variable(w_alpha*tf.random_normal([5, 5, 1, 32]))
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
	conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
	conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=0.02), train_phase, scope='bn_1')
	conv1 = tf.nn.relu(conv1)
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)
 
	w_c2 = tf.Variable(w_alpha*tf.random_normal([5, 5, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
	conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_2')
	conv2 = tf.nn.relu(conv2)
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)
 
	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
	conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_3')
	conv3 = tf.nn.relu(conv3)
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c4 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
	conv4 = batch_norm(conv4, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02), train_phase, scope='bn_4')
	conv4 = tf.nn.relu(conv4)
	conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv4 = tf.nn.dropout(conv4, keep_prob)
     
	# Fully connected layer
	w_d = tf.Variable(w_alpha*tf.random_normal([2*16*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)
 
	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out = tf.nn.softmax(out)
	# tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
	saver = tf.train.Saver()
    
    with tf.Session() as sess:

        saver.restore(sess, "crack_capcha.model-1000")
        print ("Model restored.")
       
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: imvalue ,keep_prob: 1.0}, session=sess)

def main():
	image, text, vec = obj.gen_image()
	cv2.imshow('image', image)
	cv2.waitKey(0)
	image, text, vec = obj.gen_image()
	batch_x[i,:] = image.reshape((IMAGE_HEIGHT*IMAGE_WIDTH))
	batch_y[i,:] = vec


if __name__ == '__main__':
	main()