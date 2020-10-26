import os.path
import h5py
import numpy as np
from termcolor import colored
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

SEQ_ALL = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

def common_image_name(img_name):
    if isinstance(img_name, int):
        img_index = img_name
    elif isinstance(img_name, str):
        img_index = int(img_name.split(".")[0])
    else:
        raise RuntimeError("unexpected input type @ wzy, should either like 002754 or 2754 or 002754.png")
    return img_index

def get_nr_of_channels_manually():
    # overall manual control of channels!
    NR_OF_CHANNELS = 10
    return NR_OF_CHANNELS


def cprint(text, c=3):
    color_to_choose = ['red' ,'yellow', 'green', 'cyan', 'blue', 'white', 'magenta', 'grey']
    # Nr of color        0        1       2         3       4        5        6        7
    if c > 7:
        c = c % 7
    try:
        color = color_to_choose[c]
        print(colored(text, color))
    except TypeError:
        print(text,c)
        print("^^^Error: @ wzy cprint function can't accept list, consider use () to make them as a tuple")


def read_pose_txt(filepath,seq="NO"):
    """
    return a flat_matrix with shape = (frameN, 12)
    if seq is given, return the flat_matrix of that seq in the filepath root
    """
    def _read_from_1file(path):
        flat_matrix = np.loadtxt(path, skiprows=0, comments='#')
        # print("read_pose_txt from ", os.path.relpath(path))
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


def read_pose_h5(filepath,seq="NO"):
    if seq != "NO": # seq is given
        assert os.path.isdir(filepath)
        filepath_s = os.path.join(filepath, ("xpose"+str(seq)+".h5"))
        with h5py.File(filepath_s, "r") as f:
            distance = np.array(f["distances"])
            rot_r = np.array(f["rot_r"])
            dr = np.vstack([distance, rot_r])
        return dr
    else:
        assert os.path.isfile(filepath)
        with h5py.File(filepath, "r") as f:
            distance = np.array(f["distances"])
            rot_r = np.array(f["rot_r"])
            dr = np.vstack([distance,rot_r])
        return dr


def extract_frame_dr(dr, frameID):
    assert frameID >=0
    return dr[0, frameID], dr[1, frameID]


def _euler_winkle_filter(winkle):
    """
    absolute euler winkle output limit in range +-360, 361 transfered to 1,
    which lead to problem when I subtract euler absolute results to calculate the relative winkle
    This function return e.p. -3 for winkle = 357, return 1 for winkle = -359
    """
    if abs(winkle)>180:
        if winkle > 0 :
            filtered= winkle - 360
        else:
            filtered = winkle + 360
        return filtered
    else:
        return winkle


TXTROOT_ORIGIN_ABSOLUTE_POSE_SKITTI = "/home/zwang/deeplab/research/deeplab/datasets/skitti_posestxt/"

def get_relative_xyr_pose(seq, img_name, interval_betweenF, txtroot=TXTROOT_ORIGIN_ABSOLUTE_POSE_SKITTI):
    """
    _from_absolute,
    use absolute poses of head & tail frames to calculate relative pose of them.
    if img_int - interval_betwwen_frames <0, will return d,r = 0,0

    Args:
        seq: str like "07"
        img_name: img_name: current frame's id, like "000123" or "000123.png" or "123"
        h5root: the
        interval_between_frames: = range_of_frames - 1

    Returns: x,y,beta

    """
    # locate the data

    i = common_image_name(img_name)
    # read absolute pose
        # current frame
    flat_matrix = read_pose_txt(txtroot, seq)
    matg = np.reshape(flat_matrix[i, :], (3, 4))
    x_ab = matg[0, 3]
    y_ab = matg[2, 3]
    r3x3 = R.from_matrix(matg[:, 0:3])
    rots_absulute = r3x3.as_euler('yxz', degrees=True)[0]

        # past frame, with interval
    matg_past = np.reshape(flat_matrix[i-interval_betweenF, :], (3, 4))
    x_ab_past = matg_past[0, 3]
    y_ab_past = matg_past[2, 3]
    r3x3_past = R.from_matrix(matg_past[:, 0:3])
    rots_absulute_past = r3x3_past.as_euler('yxz', degrees=True)[0]  # in radian, compatible with np.cos


    # convert to relative pose
    if i < interval_betweenF:
        x = y = r = 0
    else:
        x_re = x_ab - x_ab_past
        y_re = y_ab - y_ab_past
        dist_xy = np.sqrt(pow(x_re, 2) + pow(y_re, 2))  # output is Euclidean Distance >=0
        #### Kern tech ####
        rots_AB = np.arctan2(x_re, y_re) * 180/np.pi # radian -> degree
        rots_beta_AB = _euler_winkle_filter(rots_absulute - rots_AB)
        x = np.cos(rots_beta_AB * np.pi/180) * dist_xy  # need radian
        y = np.sin(rots_beta_AB * np.pi/180) * dist_xy

        r = _euler_winkle_filter(rots_absulute - rots_absulute_past)

    return x, y, r

def get_relative_dr_pose(seq, img_name, interval_betweenF, txtroot=TXTROOT_ORIGIN_ABSOLUTE_POSE_SKITTI):
    """
    ,_from_absolute
    use absolute poses of head & tail frames to calculate relative pose of them.
    if img_int - interval_betwwen_frames <0, will return d,r = 0,0

    Args:
        seq: str like "07"
        img_name: img_name: current frame's id, like "000123" or "000123.png" or "123"
        h5root: the
        interval_between_frames: = range_of_frames - 1

    Returns: d,r

    """
    # locate the data
    if isinstance(img_name, int):
        img_index = img_name
    elif isinstance(img_name, str):
        img_index = int(img_name.split(".")[0])
    else:
        raise RuntimeError("unexpected input type @ wzy, should either like 002754 or 2754 or 002754.png")

    # read absolute pose
        # current frame
    flat_matrix = read_pose_txt(txtroot, seq)
    i=img_index
    matg = np.reshape(flat_matrix[i, :], (3, 4))
    x_ab = matg[0, 3]
    y_ab = matg[2, 3]
    r3x3 = R.from_matrix(matg[:, 0:3])
    rots_absulute = r3x3.as_euler('yxz', degrees=True)[0]

        # past frame, with interval
    matg_past = np.reshape(flat_matrix[i-interval_betweenF, :], (3, 4))
    x_ab_past = matg_past[0, 3]
    y_ab_past = matg_past[2, 3]
    r3x3_past = R.from_matrix(matg_past[:, 0:3])
    rots_absulute_past = r3x3_past.as_euler('yxz', degrees=True)[0]

    # convert to relative pose
    if i < interval_betweenF:
        d = 0
        r = 0
    else:
        x_re = x_ab - x_ab_past
        y_re = y_ab - y_ab_past
        d = np.sqrt(pow(x_re, 2) + pow(y_re, 2))  # output is Euclidean Distance >=0

        r = _euler_winkle_filter(rots_absulute - rots_absulute_past)

    cprint("Warning wzy, dr is archived, use xyr instead",1)
    return d, r


def read_tf_record(path):
    """exp of path:/mrt/train/train-00005-of-00096.tfrecord"""
    for example in tf.python_io.tf_record_iterator(path):
        print(tf.train.Example.FromString(example))

def distribution_re_distance(h5root, seq):
    if seq =="all" or "All" or "ALL":
        dr = np.zeros((2,1))
        for seq_each in SEQ_ALL:
            dr_each = read_pose_h5(h5root, seq=seq_each)
            dr = np.hstack((dr,dr_each))
    else:
        dr = read_pose_h5(h5root, seq=seq)
    plt.figure("distribution_re_distance")
    plt.title("distribution_re_distance")
    plt.hist(dr[0,:], bins=30, color='steelblue', density=False)
    plt.figure("distribution_re_rotation")
    plt.title("distribution_re_rotation")
    plt.hist(dr[1,:], bins=30, color='orange', density=False)
    plt.show()



def trend_absolute_distance(filepath, seq="NO"):
    flat_matrix = read_pose_txt(filepath, seq=seq)
    plt.figure(2)
    plt.subplot(211)
    plt.plot(flat_matrix[:,11])
    plt.plot(flat_matrix[:,7])
    plt.plot(flat_matrix[:,3],'b')
    plt.subplot(212)
    plt.plot(flat_matrix[:,11],flat_matrix[:,3],"r.")
    plt.show()



def get_relative_pose_h5_accumulate_relative_poses(seq, img_name, h5root, interval_between_frames=1):
    """
    !!! archived !!!
    09.10 update to function "get_relative_dr_pose"

    #todo
        given the interval_between_frames, need to read the relative poses from differnet folders...

    if img_int - interval_betwwen_frames <0, will return d,r = 0,0
    :param seq: str like "07"
    :param img_name: current frame's id, like "000123" or "000123.png" or "123"
    :param h5root: path to h5 files' root folder (it stores relative poses)
    :return: the d,r pose, with the given img_name as the current frame
    """

    dr = read_pose_h5(h5root,seq)

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
            add_d ,add_r = extract_frame_dr(dr, relative_pose_index - interval)
            d += add_d
            r += add_r
    return d, r


# -----------for validations --------------
"""
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
    H5ROOT = "/home/zwang/deeplab/research/deeplab/datasets/skitti_xxr_pose_h5_intr2"
    seq = "01"
    img_names = ["000000.png","000001.png", "000002.png","000003.png","000004.png","000101.png"]
    for img_name in img_names:
        print("\n", img_name)
        print("1", get_relative_pose_h5_accumulate_relative_poses(seq, img_name, H5ROOT, interval_between_frames=1))
        #print("2",get_relative_pose_h5_accumulate_relative_poses(seq,img_name,H5ROOT,2))
        #print("3",get_relative_pose_h5_accumulate_relative_poses(seq,img_name,H5ROOT,3))


def varify_if_negative_re_distance():
    H5ROOT = "/home/zwang/deeplab/research/deeplab/datasets/skitti_xxr_pose_h5_intr2"
    seq = "02"
    print("running...")
    for i in range(200,1000):
        img_int = "{number:06}".format(number=100)
        img_name = img_int + ".png"
        d,r = get_relative_pose_h5_accumulate_relative_poses(seq, img_name, H5ROOT, interval_between_frames=1)
        assert d>0
    print("no negative re_distance in {}".format(seq))
"""
if __name__ == '__main__':
    TXTROOT="/home/zwang/deeplab/research/deeplab/datasets/skitti_posestxt"
    H5ROOT = "/home/zwang/deeplab/research/deeplab/datasets/skitti_xxr_pose_h5_intr2"

    #read_tf_record("/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_3frames/train-00006-of-00096.tfrecord")
    #trend_absolute_distance(TXTROOT, "01")
    #distribution_re_distance(H5ROOT, "all")
    #print(get_relative_dr_pose("04",123,11))
    print(get_relative_xyr_pose("04",123,11))

