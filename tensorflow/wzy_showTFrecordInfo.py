import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from PIL import Image
import cv2



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
    #dataset = tf.data.TFRecordDataset(filenames=[path_to_file])
    #raw_example = next(iter(dataset))
    raw_dataset = tf.data.TFRecordDataset(path_to_file)

    #parsed = tf.train.Example.FromString(raw_example.numpy())
    #out= parsed.features.feature['image/det_z_min/encoded']
    #cprint(type(out))

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
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)
    parsed_dataset = raw_dataset.map(_parse_function)
    cprint(type(parsed_dataset))
    for parsed_record in parsed_dataset.take(1):   # each is a dict
        cprint(parsed_record.keys())
        height = parsed_record['image/height']
        width = parsed_record['image/width']
        observation=parsed_record['image/observations/encoded']
        de_obser = tf.image.decode_png(observation).numpy()
        print(height.numpy(), width.numpy(), type(de_obser), np.shape(de_obser))

        cv2.imwrite('/home/zwang/Downloads/{}.png'.format("obser"), de_obser)


if __name__ == '__main__':
    tf.enable_eager_execution()
    TFPATH = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/val/val-00000-of-00021.tfrecord"
    #simple_print_content_of_TFrecord("/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange20/val/val-00000-of-00021.tfrecord")
    print_TFrecordimage_tf2fashion(TFPATH)
    pass