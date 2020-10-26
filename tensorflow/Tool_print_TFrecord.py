import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from PIL import Image
import cv2

###
# must be used in tensorflow 2.0 +
###

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

def print_TFrecordimage(path_to_file, combine_image_Amount=1):

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    def _save_to_image(key:str,range_oftfrecord,i):
        """
        z_max = parsed_record['image/det_z_max/encoded']
        file_path = '/home/zwang/Downloads/{}_{}_{}.png'.format(range_oftfrecord, i, "z_max")
        cv2.imwrite(file_path, tf.image.decode_png(z_max).numpy())
        print("output into {}".format(file_path))
        """
        tmp_img = parsed_record[key]
        file_path = '/tmp/{}_{}_{}_{}.png'.format(range_oftfrecord, i, key.split("/")[-2], key.split("/")[-3])
        cv2.imwrite(file_path, tf.image.decode_png(tmp_img).numpy())
        print("output into {}".format(file_path))

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
    if combine_image_Amount==2:
        feature_description_plus2= {
        'image/pre_frame/det_z_min/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/pre_frame/det_z_max/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/pre_frame/intensity/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/pre_frame/observations/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/pre_frame/observ_z_min/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        #'image/re_rotation': tf.io.FixedLenFeature((), tf.float32, default_value=''),
        }
        feature_description.update(feature_description_plus2)
    if combine_image_Amount==3:
        cprint("not implemented jet")
        return None

    parsed_dataset = raw_dataset.map(_parse_function)
    range_oftfrecord = path_to_file.split('/')[-3]
    for i, parsed_record in enumerate(parsed_dataset.take(3)):   # each is a dict
        _save_to_image('image/det_z_max/encoded',range_oftfrecord,i)
        _save_to_image('image/segmentation/class/encoded_sparse',range_oftfrecord,i)
        #_save_to_image('image/segmentation/class/encoded_dense',range_oftfrecord,i)

        if combine_image_Amount==2:
            _save_to_image('image/pre_frame/det_z_max/encoded',range_oftfrecord,i)


if __name__ == '__main__':
    tf.enable_eager_execution()
    #TFPATH_RANGE10_VAL = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/val/val-00000-of-00021.tfrecord"
    TF_RANGE2_comb_TR = "/mrtstorage/users/zwang/pcd_mapper_pastonly/combine_12/train/train-00005-of-00096.tfrecord"
    TF_BASELINE_INT4 = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval4/val/val-00000-of-00021.tfrecord"
    #simple_print_content_of_TFrecord("/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange20/val/val-00000-of-00021.tfrecord")
    print_TFrecordimage(TF_BASELINE_INT4, combine_image_Amount=1)
