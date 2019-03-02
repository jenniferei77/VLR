from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime
import csv
from matplotlib import pyplot
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import util

import pdb

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class CaffeNet(keras.Model):
    def __init__(self, num_classes=10):
        super(CaffeNet, self).__init__(name='CaffeNet')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=96,
                                   kernel_size=[11, 11],
                                   strides=(4, 4),
                                   padding="valid",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=(2, 2))
        self.conv2 = layers.Conv2D(filters=256,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=(2, 2))
        self.conv3 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(3, 3),
                                      strides=(2, 2))
        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        #out = self.dropout2(out, training=training)
        #out = self.dense3(out)
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
    return image, label, weight

def _parse_test_function(image, label, weight):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, 0.875)
    image = image - [[123.68, 115.78, 103.94]]
    return image, label, weight
    
def task5(restore_dir, data_dir, model, test_dataset):
    ## Task 5 Analysis:
    images_to_eval = {'boat':'000069', 'bottle':'000202', 'bus':'000252', 'car':'000172', 'cat':'000292', 'chair':'000008', 'cow':'000025', 'dog':'000029', 'horse':'000010', 'person':'000010'}
    print('Done evaluating the 10 images')
    class_to_mat = util.load_test_images(images_to_eval, data_dir) 
    conv1_outputs = np.zeros([1,20])
    pool3_outputs = np.zeros([1,20])
    iterator = 0
    batch = 20
    images = np.zeros([])
    for batch, (image, labels, weights) in enumerate(test_dataset):
        iterator += 1
        print(image.shape)
        #image = np.expand_dims(image, axis=0).astype('float32')
        conv1_output = np.asarray(model.conv1(image))
        pool3_output = np.asarray(model.pool3(image))
        if iterator == 1:
            images = np.asarray(image)
            conv1_outputs = conv1_output
            pool3_outputs = pool3_output
        else:
            images = np.concatenate((images, np.asarray(image)), axis=0)
            conv1_outputs = np.concatenate((conv1_outputs, conv1_output), axis=0)
            pool3_outputs = np.concatenate((pool3_outputs, pool3_output), axis=0)
    print('Done getting outputs')
    pdb.set_trace()
    conv1_outputs = (conv1_outputs.reshape(conv1_outputs.shape[0], conv1_outputs.shape[1]*conv1_outputs.shape[2]*conv1_outputs.shape[3]))
    pool3_outputs = pool3_outputs.reshape(4952, 36963)
    pdb.set_trace()

    #conv1_tree = cKDTree(conv1_outputs)
    #pool3_tree = cKDTree(pool3_outputs)
    print('Done making trees')
    conv1_closest = {}
    pool3_closest = {}
    
    for key in class_to_mat:
        conv1_out = np.asarray(model.conv1(tf.convert_to_tensor(class_to_mat[key])))
        pool3_out = np.asarray(model.pool3(tf.convert_to_tensor(class_to_mat[key])))
        conv1_norms = np.linalg.norm(conv1_out - conv1_outputs)
        pool3_norms = np.linalg.norm(pool3_out - pool3_outputs)
        conv1_ind = np.argpartition(conv1_norms, 5)[:5]
        pool3_ind = np.argpartition(pool3_norms, 5)[:5]
        #conv1_dist, conv1_ind = conv1_tree(conv1_out, k=5)
        #pool3_dist, pool3_ind = pool3_tree(pool3_out, k=5)
        pdb.set_trace()
        conv1_closest[class_to_mat[key]] = images[conv1_ind]
        fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = pyplot.subplots(1,6)
        ax1.imshow(class_to_mat[key])
        ax2.imshow(images[conv1_ind[0]])
        ax3.imshow(images[conv1_ind[1]])
        ax4.imshow(images[conv1_ind[2]])
        ax5.imshow(images[conv1_ind[3]])
        ax6.imshow(images[conv1_ind[4]])
        pyplot.show()
        pyplot.savefig('/data/VLR/hw1/images_dir/' + key + '_conv1_KNNs.png')
        
        pool3_closest[class_to_mat[key]] = images[pool3_ind]
        fig2, (ax7, ax8, ax9, ax10, ax11, ax12) = pyplot.subplots(1,6)
        ax7.imshow(class_to_mat[key])
        ax8.imshow(images[pool3_ind[0]])
        ax9.imshow(images[pool3_ind[1]])
        ax10.imshow(images[pool3_ind[2]])
        ax11.imshow(images[pool3_ind[3]])
        ax12.imshow(images[pool3_ind[4]])
        pyplot.show()
        pyplot.savefig('/data/VLR/hw1/images_dir/' + key + '_pool3_KNNs.png')

        pdb.set_trace()
    
    return conv1_closest, pool3_closest

#def task5_3(test_dataset, model):
    
    

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval_interval', type=int, default=250,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log_dir', type=str, default='03_output',
                        help='path for logging directory')
    parser.add_argument('--data_dir_train', type=str, default='./VOCtrain/VOC2007',
                        help='Path to PASCAL train data storage')
    parser.add_argument('--data_dir_test', type=str, default='./VOCtest/VOC2007',
                        help='Path to PASCAL test data storage')
    parser.add_argument('--train_stage1', type=str, default='/data/hw1/03_output/2019-03-01_17-05-30/training_checkpoints/', help='Path to checkpoint 1')

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

    randomize = np.arange(1000)
    shuffled_images = test_images[randomize]
    shuffled_labels = test_labels[randomize]
    shuffled_weights = test_labels[randomize]
    post_dataset = tf.data.Dataset.from_tensor_slices((shuffled_images, shuffled_labels, shuffled_weights))
    pdb.set_trace()
    
    model = CaffeNet(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
        
    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    
    start_lr = args.lr
    decayed_lr = tf.train.exponential_decay(start_lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(decayed_lr(), 0.9)
    
    checkpoint_path = logdir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint_inc = int(args.epochs*(train_labels.shape[0]/args.batch_size)/3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step_counter=global_step)    

    checkpoint_counter = 0
    acc_count = 0
    train_log = {'iter': [], 'loss': [], 'mAP': []}
    test_log = {'iter': [], 'loss': [], 'mAP': []}
    test_mAP = 0
 
    init_AP, init_mAP = util.eval_dataset_map(model, test_dataset)
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('test_mAP', init_mAP)
    
    checkpoint.restore(args.train_stage1 + 'ckpt2-3.index')    
 
# Task 5.3   
    #post_outputs = np.empty([])
    #post_iter = 0
    #pdb.set_trace() 
    #batch = 5
    #for batch, (image, labels, weights) in enumerate(post_dataset):
    #    post_iter += 1
    #    image = tf.expand_dims(image, axis=0)
    #    labels = tf.expand_dims(image, axis=0)
    #    weights = tf.expand_dims(weights, axis=0)
    #    post_output = np.asarray(model.dense2(image))
    #    if pose_iter == 1:
    #        post_outputs = post_output
    #    else:
    #        post_outputs = np.concatenate((post_outputs, post_output), axis=0)

    #post_embedded = TSNE(n_components=1000).fit_transform(post_outputs)
    #color_map = np.linspace(0.0, 1.0, num=20)
    #post_labels = np.multiply(shuffled_labels, color_map)
    #colors = np.sum(post_labels, axis=1)
    #fig_tsne, ax_tsne = pyplot.scatter(post_embedded[:, 0], post_embedded[:, 1], c=colors, cmap=pyplot.cm.Spectral)
 
    #pyplot.savefig('/data/VLR/hw1/images_dir/' + 'tsne.png')
    
 ## Task 5.1
    #filter1 = np.asarray(model.conv1.trainable_weights[0][:,:,1,1])
    #w_min = np.min(filter1)
    #w_max = np.max(filter1)
    
    #pyplot.imshow(filter1, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
    #pyplot.show()
    #pyplot.savefig(args.train_stage1 + '/filter3.png')
 
    #conv1_closest, pool3_closest = task5(args.train_stage1, args.data_dir_test, model, test_dataset) 
    
    for epoch in range(args.epochs):
        total_loss = 0
        #total_mAP = 0
        for batch, (image, label, weight) in enumerate(train_dataset):
            current_loss, gradients = util.cal_grad(model, loss_func=tf.losses.sigmoid_cross_entropy, inputs=image, targets=label)
            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('current_loss', current_loss)
            print('current loss: ', current_loss.numpy())   
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step)
            total_loss += current_loss
            loss_avg = total_loss/global_step.numpy()
            if global_step.numpy() % args.log_interval == 0:
                output_labels = tf.nn.sigmoid(model(image, training=False))
                acc_count += 1
                train_mAP = np.nanmean(np.asarray(util.compute_ap(label.numpy(), output_labels.numpy(), weight.numpy(), average=none)))
                print('epoch: {0:d}/{1:d} step:{2:d} training loss:{3:.4f} training accuracy:{4:.4f}'.format(epoch,
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
                
                print('global step:{0:d}, test loss:{1:.4f}, test mAP:{2:.4f}'.format(global_step.numpy(), final_test_loss, test_mAP))
                test_log['iter'].append(global_step.numpy())
                test_log['loss'].append(final_test_loss)
                test_log['mAP'].append(test_mAP)
                with writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('final_test_loss', final_test_loss)
                    tf.contrib.summary.scalar('test_mAP', test_mAP)
            if global_step.numpy() % checkpoint_inc == 0:
                checkpoint.save(file_prefix=checkpoint_prefix + str(checkpoint_counter))
                checkpoint_counter += 1    
    
    for cid, cname in enumerate(class_names):
        print('{}: {}'.format(cname, util.get_el(test_ap, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
