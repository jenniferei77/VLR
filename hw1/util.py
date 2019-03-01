import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
import cv2
import pdb


def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    ## TODO Implement this function
    first_class_file = data_dir + "/ImageSets/Main/" + class_names[0] + '_' + split + ".txt"
    with open(first_class_file, 'r') as f:
        lines = f.read().splitlines()
        images_count = len(lines)
    images = np.zeros([images_count, 256, 256, 3], dtype=np.float32)
    labels = np.zeros([images_count, len(class_names)], dtype=np.int32)
    weights = np.zeros([images_count, len(class_names)], dtype=np.int32)
    load_images = True
    for class_name in class_names:
        #Access trainval file for each class
        image_names_with_labels = data_dir + "/ImageSets/Main/" + class_name + '_' + split + ".txt"
        with open(image_names_with_labels, 'r') as f:
            line_number = 0
            #Iterate through each image of the trainval class file
            for line in f:
                #Split lines into image name and label
                line_split = line.split(' ')
                image_name = line_split[0]
                image_label = int(line_split[-1][:-1])
                if image_label == 1:
                    labels[line_number, class_names.index(class_name)] = int(1)
                    weights[line_number, class_names.index(class_name)] = int(1)
                elif image_label == 0:
                    labels[line_number, class_names.index(class_name)] = int(1)
                    weights[line_number, class_names.index(class_name)] = int(0)
                else: 
                    labels[line_number, class_names.index(class_name)] = int(0)
                    weights[line_number, class_names.index(class_name)] = int(1)

                #Access image from image_name and resize image to 256x256x3    
                if load_images:
                    image = cv2.imread(data_dir + "/JPEGImages/" + image_name + ".jpg")
                    image_resize = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    image_resize = np.expand_dims(image_resize, axis=0)
                    images[line_number, :, :, :] = image_resize
                line_number += 1
            load_images = False
    return images, labels, weights

def load_test_images(images_to_eval, data_dir):
    class_to_mat = {}
    for key, value in images_to_eval:
        image = cv2.imread(data_dir + "/JPEGImages/" + value + ".jpg")
        image_resize = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        image_dict[key] = image_resize
    return class_to_mat

def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP

def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    ## TODO implement the code here
    iterator = 0
    labels = np.empty([])
    weights = np.empty([])
    output_labels = np.empty([])
    for (batch, (image, label, weight)) in enumerate(dataset):
        iterator += 1
        output_label = tf.nn.sigmoid(model(image, training=False))

        if iterator == 1:
            labels = np.asarray(label)
            weights = np.asarray(weight)
            output_labels = np.asarray(output_label).astype('float32')
        else:
            labels = np.concatenate((labels, label), axis=0)
            weights = np.concatenate((weights, weight), axis=0)
            output_labels = np.concatenate((output_labels, output_label), axis=0).astype('float32')

    AP = compute_ap(labels, output_labels, weights, average=None)

    mAP = np.nanmean(AP).astype('float32')

    return AP, mAP


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr
