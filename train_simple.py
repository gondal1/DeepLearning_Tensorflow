#====================================
# Author: Muhammad Waleed Gondal
# Date: 11, December, 2016
#====================================
import tensorflow as tf
import numpy as np
from deep_model import Model
#from util import augment
import os
import matplotlib.pyplot as plt
import csv
import time
#================================================
# Setting Hyper-Parameters 
#================================================

PRETRAINED_MODEL_PATH= None 
N_EPOCHS = 300
INIT_LEARNING_RATE = 0.01
WEIGHT_DECAY_RATE = 0.0005
MOMENTUM = 0.9
IMAGE_HEIGHT  = 256    #960
IMAGE_WIDTH   = 256    #720
NUM_CHANNELS  = 3
BATCH_SIZE = 100
N_CLASSES = 2
DROPOUT = 0.50
ckpt_dir = "./ckpt_dir"
LOGS_PATH = './tensorflow_logs'
WEIGHT_PATH = '.npy'
TRAINSET_PATH = '.csv'
VALSET_PATH ='.csv'


#=======================================================================================================
# Reading Training data from CSV FILE
#=======================================================================================================

csv_path = tf.train.string_input_producer([TRAINSET_PATH], shuffle=True)
textReader = tf.TextLineReader()
_, csv_content = textReader.read(csv_path)
im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

im_content = tf.read_file(im_name)
train_image = tf.image.decode_jpeg(im_content, channels=3)
train_image = tf.cast(train_image, tf.float32) / 255. # necessary for mapping rgb channels from 0-255 to 0-1 float. 
#train_image = augment(train_image)
size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH], tf.int32)
train_image = tf.image.resize_images(train_image, size)
train_label = tf.cast(im_label, tf.int64) # unnecessary
train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=BATCH_SIZE,
                                                             capacity = 1000 + 3*BATCH_SIZE, min_after_dequeue = 1000)

#=======================================================================================================
# Reading Validation data from CSV FILE
#=======================================================================================================

val_csv_path = tf.train.string_input_producer([VALSET_PATH], shuffle=True)
val_textReader = tf.TextLineReader()
_, val_content = val_textReader.read(val_csv_path)
val_image, val_label = tf.decode_csv(val_content, record_defaults=[[""], [1]])

val_image_content = tf.read_file(val_image)
val_image = tf.image.decode_jpeg(val_image_content, channels=3)
val_image = tf.cast(val_image, tf.float32) / 255. # necessary
size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH], tf.int32)
val_image = tf.image.resize_images(val_image, size)
val_label = tf.cast(val_label, tf.int64) # unnecessary
val_image_batch, val_label_batch = tf.train.shuffle_batch([val_image, val_label], batch_size=BATCH_SIZE,
                                                         capacity = 1000 + 3*BATCH_SIZE, min_after_dequeue = 1000)

#=======================================================================================================
# Placeholders for feeding data 
#=======================================================================================================
learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels') 

#==============================================================================================================
# Defining object for the model class, send initialization weights and variables in here if defined in class.
#==============================================================================================================
detector = Model(N_CLASSES)

#==============================================================================================================
# Retrieving output from the class defined, in most cases it would be the prediction.
#==============================================================================================================
output = detector.inference(images_tf, DROPOUT)

#==============================================================================================================
# Defining Loss, could be changed from cross entropy depending on needs. The current configuration works well on 
# multiclass (not hot-encoded vectors) prediction like ImageNET.
#==============================================================================================================
with tf.name_scope('Loss'):
    loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( output, labels_tf ), name='loss_tf')
    loss_summary = tf.scalar_summary("loss", loss_tf)
    weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
    weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * WEIGHT_DECAY_RATE
    loss_tf += weight_decay
    
#==============================================================================================================
# Optimizer, again it can be changed to any function provided by Tensorflow. You can simply use commented out line
# instead of explicitly computing gradients, if you are not interested in creating summaries of gradients.
#==============================================================================================================
#train_op = tf.train.MomentumOptimizer( learning_rate, MOMENTUM).minimize(loss_tf)
optimizer = tf.train.MomentumOptimizer( learning_rate, MOMENTUM)
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), 
                     grads_and_vars)
grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )

#===================================================================================================================
# Summaries for the gradients
#===================================================================================================================
for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)
summary_op = tf.merge_all_summaries()

#===================================================================================================================
# Accuracy for the current batch
#===================================================================================================================
correct_pred = tf.equal(tf.argmax(output, 1), labels_tf)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#===================================================================================================================
# Saver Operation to save and restore all variables.
#===================================================================================================================
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(max_to_keep=50)



with tf.Session() as sess:
    
    if PRETRAINED_MODEL_PATH:
        print "Using Pretrained model"
        saver.restore(sess, PRETRAINED_MODEL_PATH)
    else:    
        sess.run(tf.initialize_all_variables())
    
    # For populating queues with batches, very important!
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Following Lists are defined for storing important training & val information, for later matplotlib visualization. 
    loss_list, plot_trainAccuracy, plot_loss, plot_valAccuracy = [], [], [], []
    summary_writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
    steps = 1
    count = 1
    
    for epoch in range(N_EPOCHS):    
        
        train_correct = 0
        train_data = 0    
        epoch_start_time = time.time()
        
        for i in range (trainset/BATCH_SIZE +1):      # You can simply put a number here, but you definitely need to know the                                                         # size of training set. 
            train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
            _, loss_val, output_val, train_accuracy, summary = sess.run([train_op, loss_tf, output, accuracy, summary_op], 
                                                                  feed_dict={learning_rate: INIT_LEARNING_RATE, 
                                                                             images_tf: train_imbatch, labels_tf:
                                                                             train_labatch})
            
            loss_list.append(loss_val)                               
            plot_trainAccuracy.append(train_accuracy)                           # For visualizing training accuracy curve
            plot_loss.append(loss_val)                                     # For visualizing training loss curve
            
            train_data += len(output_val)
            
            if (steps) % 5 == 0:   # after 5 batches
                print "======================================"
                print "Epoch", epoch+1, "Iteration", steps
                print "Processed", train_data, '/', trainset               # (count*BATCH_SIZE)
                print 'Accuracy: ', train_accuracy
                print "Training Loss:", np.mean(loss_list)
                loss_list = []
                summary_writer.add_summary(summary, steps)
            steps += 1
            count += 1
        count = 1
        
        
        for i in range (testset/BATCH_SIZE +1):
            
            val_imbatch, val_labatch = sess.run([val_image_batch, val_label_batch])
            val_accuracy = sess.run(accuracy, feed_dict={images_tf:val_imbatch, labels_tf: val_labatch})
            
        f_log.write('epoch:'+str(epoch+1)+'\tacc:'+str(val_accuracy) + '\n')
        print "===========**VALIDATION ACCURACY**================"
        print 'epoch:'+str(epoch+1)+'\tacc:'+str(val_accuracy) + '\n'
        print 'Time Elapsed for Epoch:'+str(epoch+1)+' is '+str ((time.time() - epoch_start_time)/60.)+' minutes'
        plot_valAccuracy.append(val_accuracy)
        INIT_LEARNING_RATE *= 0.99
        
        # Saving Weights after each 10 epochs
        if (epoch % 10 == 0):            
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=epoch)
    