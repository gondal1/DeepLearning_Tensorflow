import tensorflow as tf
import numpy as np
import cPickle

class Model():
    def __init__(self, n_labels):
        self.image_mean = [103.939, 116.779, 123.68] # Imagenet Mean Values for R, G and B
        self.n_labels = n_labels

    
    def new_conv_layer( self, bottom, filter_shape, name ):
        with tf.variable_scope( name, reuse=True) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

        return bias #relu


    def new_fc_layer( self, bottom, input_size, output_size, name ):
        shape = bottom.get_shape().to_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])

        with tf.variable_scope(name, reuse=True) as scope:
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b, name=scope)

        return fc

    def inference( self, rgb, train=False ):
        rgb *= 255.
        r, g, b = tf.split(3, 3, rgb)
        bgr = tf.concat(3,
            [
                b-self.image_mean[0],
                g-self.image_mean[1],
                r-self.image_mean[2]
            ])

        relu1_1 = self.new_conv_layer( bgr, [3,3,3,64], "conv1_1" )
        relu1_2 = self.new_conv_layer( relu1_1, [3,3,64,64], "conv1_2" )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')

        relu2_1 = self.new_conv_layer(pool1, [3,3,64,128],"conv2_1")
        relu2_2 = self.new_conv_layer(relu2_1,[3,3,128,128],"conv2_2")
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = self.new_conv_layer( pool2, [3,3,128,256], "conv3_1")
        relu3_2 = self.new_conv_layer( relu3_1, [3,3,256,256], "conv3_2")
        relu3_3 = self.new_conv_layer( relu3_2, [3,3,256,256], "conv3_3")
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

        relu4_1 = self.new_conv_layer( pool3, [3,3,256,512], "conv4_1")
        relu4_2 = self.new_conv_layer( relu4_1, [3,3,512,512],"conv4_2")
        relu4_3 = self.new_conv_layer( relu4_2, [3,3,512,512], "conv4_3")
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

        relu5_1 = self.new_conv_layer( pool4, [2,2,512,512], "conv5_1")
        relu5_2 = self.new_conv_layer( relu5_1, [2,2,512,512], "conv5_2")
        relu5_3 = self.new_conv_layer( relu5_2, [2,2,512,512], "conv5_3")
        pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
        
        
        fc1 = self.new_fc_layer( pool5, 1024, 1024, 'fc1')

        
        with tf.variable_scope("FC2"):
            fc2 = tf.get_variable(
                    "W",
                    shape=[1024, self.n_labels],
                    initializer=tf.random_normal_initializer(0., 0.01))

        prediction = tf.matmul( fc1,fc2)

        return prediction






