import os.path
import h5py
import numpy as np
from termcolor import colored
import tensorflow as tf



SEQ_ALL = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


def cprint(text, c=3):
    color_to_choose = ['red' ,'yellow', 'green', 'cyan', 'blue', 'white', 'magenta', 'grey']
    # Nr of color        0        1       2         3       4        5        6        7
    if c>7: c = c % 7
    color = color_to_choose[c]
    print(colored(text, color))
    return

def read_pose_txt(filepath,seq="NO"):
    """
    return a flat_matrix with shape = (frameN, 12)
    if seq is given, return the flat_matrix of that seq in the filepath root
    """
    def _read_from_1file(path):
        flat_matrix = np.loadtxt(path, skiprows=0, comments='#')
        print("read_pose_txt from ", os.path.relpath(path))
        return flat_matrix

    if seq != "NO": # seq is given
        assert os.path.isdir(filepath)
        filepath_s = os.path.join(filepath, (str(seq)+".txt"))
        return _read_from_1file(filepath_s)
    elif seq =="NO":
        assert os.path.isfile(filepath)
        return _read_from_1file(filepath)
    else:
        raise RuntimeError("wrong filepath")


def read_pose_h5(filepath):
    #print("read_pose_h5 from " , filepath)
    assert os.path.isfile(filepath)
    with h5py.File(filepath, "r") as f:
        distance = np.array(f["distances"])
        rot_r = np.array(f["rot_r"])
        dr = np.vstack([distance,rot_r])
    return dr


def get_relative_pose_h5(seq, img_name, pose_root, interval_between_frames=1):
    """
    if img_int - interval_betwwen_frames <0, will return d,r = 0,0
    :param seq: str like "07"
    :param img_name: current frame's id, like "000123" or "000123.png" or "123"
    :param pose_root: path to h5 files' root folder
    :return: the d,r pose, with the given img_name as the current frame
    """
    # h5 file path
    h5file_name= "xpose"+seq+".h5"
    filepath = os.path.join(pose_root,h5file_name)
    dr = read_pose_h5(filepath)
    # locate the data
    if isinstance(img_name,int):
        img_index = img_name
    elif isinstance(img_name,str):
        img_index = int(img_name.split(".")[0])
    else:
        raise RuntimeError("unexpected input type @ wzy")
    relative_pose_index = img_index - 1  # relative pose's index is 1 less than the image_index
    # we store the Relative pose to the past Frame in the current Frame -> R0=F1-F0
    # F0 F1 F2 F3 - frames
    #    R0 R1 R2 - relative poses
    if relative_pose_index>=0:
        d,r = extract_frame_dr(dr, relative_pose_index) # base: relative pose to last frame
    else:  d,r =0,0
    if interval_between_frames >= 2: # add more relative poses
        for interval in range(1, interval_between_frames):
            # for exp, interval = 3, need to add R_0, R_-1, R_-2
            # R_0 is the base of d,r; in this loop we add R_-1 and R_-2
            add_d ,add_r = extract_frame_dr(dr, relative_pose_index -interval)
            d += add_d
            r += add_r
    return d, r


def extract_frame_dr(dr, frameID):
    assert frameID >=0
    return dr[0, frameID], dr[1, frameID]

def read_tf_record(path):
    """exp of path:/mrt/train/train-00005-of-00096.tfrecord"""
    for example in tf.python_io.tf_record_iterator(path):
        print(tf.train.Example.FromString(example))
        break

# -----------for validations --------------
def compare_two_txt_pose_source():
    m1 = read_pose_txt("/home/zwang/multimodal-deeplab/research/deeplab/datasets/skitti_posestxt/07.txt")
    m2 = read_pose_txt("/home/zwang/multimodal-deeplab/research/deeplab/datasets/old_posestxt/07.txt")
    print(type(m1),m1.shape)
    frameN,t = m1.shape
    n=0
    for i in range(frameN):
        for k in range(t):
            if abs(m1[i,k]-m2[i,k])>0.2:
                n=n+1
                print(m1[i,k],m2[i,k])
    print(n,n/frameN/t)


def varify_get_relative_pose_h5():
    pose_root = "/home/zwang/multimodal-deeplab/research/deeplab/datasets/skitti_xxr_pose_h5"
    seq = "04"
    img_names = ["000000.png","000001.png", "000002.png","000003.png","000004.png","000101.png"]
    for img_name in img_names:
        print("\n",img_name)
        print("1",get_relative_pose_h5(seq,img_name,pose_root,1))
        print("2",get_relative_pose_h5(seq,img_name,pose_root,2))
        print("3",get_relative_pose_h5(seq,img_name,pose_root,3))



if __name__ == '__main__':
    read_tf_record("/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_3frames/train-00006-of-00096.tfrecord")