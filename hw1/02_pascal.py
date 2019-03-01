from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime
import csv

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import util

import pdb

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout = layers.Dropout(rate=0.4)
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def _parse_train_function(image, label, weight):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = image - [[123.68, 116.78, 103.94]]
    image = image / 255.0 
    return image, label, weight

def _parse_test_function(image, label, weight):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, 0.875)
    image = image - [[123.68, 115.78, 103.94]]
    image = image / 255.0
    return image, label, weight

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=50,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log_dir', type=str, default='02_output',
                        help='path for logging directory')
    parser.add_argument('--data_dir_train', type=str, default='./VOCtrain/VOC2007',
                        help='Path to PASCAL train data storage')
    parser.add_argument('--data_dir_test', type=str, default='./VOCtest/VOC2007',
                        help='Path to PASCAL test data storage')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir_train,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir_test,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    ## TODO modify the following code to apply data augmentation here
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    train_dataset = train_dataset.map(_parse_train_function)
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(_parse_test_function)
    test_dataset = test_dataset.shuffle(10000).batch(args.batch_size)

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
        
    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    acc_count = 0
    train_log = {'iter': [], 'loss': [], 'mAP': []}
    test_log = {'iter': [], 'loss': [], 'mAP': []}
    test_mAP = 0

    init_AP, init_mAP = util.eval_dataset_map(model, test_dataset)
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('test_mAP', init_mAP)
       

    for epoch in range(args.epochs):
        total_loss = 0
        #total_mAP = 0
        for batch, (image, label, weight) in enumerate(train_dataset):
            current_loss, gradients = util.cal_grad(model, loss_func=tf.losses.sigmoid_cross_entropy, inputs=image, targets=label)
            print('Current Loss: ', current_loss.numpy())   
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step)
            total_loss += current_loss
            loss_avg = total_loss/global_step.numpy()
            if global_step.numpy() % args.log_interval == 0:
                output_labels = tf.nn.sigmoid(model(image, training=False))
                acc_count += 1
                train_mAP = np.nanmean(np.asarray(util.compute_ap(label.numpy(), output_labels.numpy(), weight.numpy(), average=None)))
                print('Epoch: {0:d}/{1:d} Step:{2:d} Training Loss:{3:.4f} Training Accuracy:{4:.4f}'.format(epoch,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         current_loss,
                                                         train_mAP))
                with writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('training_loss', current_loss)
                    tf.contrib.summary.scalar('training_mAP', train_mAP)
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(current_loss)
                train_log['mAP'].append(train_mAP)
            if global_step.numpy() % args.eval_interval == 0:
                total_test_loss = 0
                test_inc = 0
                for test_batch, (test_image, test_label, test_weight) in enumerate(test_dataset):
                    test_loss = tf.losses.sigmoid_cross_entropy(test_label, model(test_image, training=False), test_weight)
                    test_inc += 1
                    total_test_loss += test_loss
                final_test_loss = total_test_loss/test_inc
                test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                
                print('Global Step:{0:d}, Test Loss:{1:.4f}, Test mAP:{2:.4f}'.format(global_step.numpy(), final_test_loss, test_mAP))
                test_log['iter'].append(global_step.numpy())
                test_log['loss'].append(final_test_loss)
                test_log['mAP'].append(test_mAP)
                with writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('final_test_loss', final_test_loss)
                    tf.contrib.summary.scalar('test_mAP', test_mAP)
    with open(logdir + '.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        for key, value in train_log.items():
          csv_file.writerow([key, value])
        for key, value in test_log.items():
          csv_file.writerow([key, value])
    
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(test_AP, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
