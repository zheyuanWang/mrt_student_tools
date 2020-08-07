import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from PIL import Image
import cv2
import os


def cprint(text, c=3):
    color_to_choose = ['red' ,'yellow', 'green', 'cyan', 'blue', 'white', 'magenta', 'grey']
    # Nr of color        0        1       2         3       4        5        6        7
    if c>7: c = c % 7
    color = color_to_choose[c]
    print(colored(text, color))
    return

def simple_print_content_of_TFrecord(path):
    """https://stackoverflow.com/questions/42394585/how-to-inspect-a-tensorflow-tfrecord-file
    """
    for example in tf.python_io.tf_record_iterator(path):
        print(tf.train.Example.FromString(example))
        break

def print_TFrecordimage_tf2fashion(path_to_file):
    raw_dataset = tf.data.TFRecordDataset(path_to_file)
    feature_description = {
        'image/det_z_min/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/det_z_max/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/intensity/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/observations/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/observ_z_min/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
        'image/channels':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded_sparse':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/encoded_dense':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
    }
    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)
    parsed_dataset = raw_dataset.map(_parse_function)
    range_oftfrecord = path_to_file.split('/')[-3]
    for i, parsed_record in enumerate(parsed_dataset.take(3)):   # each is a dict
        # input image z_max
        z_max = parsed_record['image/det_z_max/encoded']
        cv2.imwrite('/home/zwang/Downloads/{}_{}_{}.png'.format(range_oftfrecord,i, "z_max"),
                    tf.image.decode_png(z_max).numpy())
        # label as image
        label_sparse = parsed_record['image/segmentation/class/encoded_sparse']
        de_label = tf.image.decode_png(label_sparse).numpy()
        cv2.imwrite('/home/zwang/Downloads/{}_{}_{}.png'.format(range_oftfrecord,i, "label"), de_label)

if __name__ == '__main__':
    tf.enable_eager_execution()
    TFPATH_RANGE10_VAL = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/val/val-00000-of-00021.tfrecord"
    TFPATH_RANGE5_TRAIN = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange5+1/train/train-00003-of-00096.tfrecord"

    #simple_print_content_of_TFrecord("/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange20/val/val-00000-of-00021.tfrecord")
    print_TFrecordimage_tf2fashion(TFPATH_RANGE5_TRAIN)
    pass