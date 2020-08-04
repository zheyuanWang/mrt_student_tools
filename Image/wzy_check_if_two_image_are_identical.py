import cv2
import numpy as np
import os
import shutil
import glob

def compare_via_glob(dir1,dir2):
    list1 = glob.glob(dir1+"/*.png")
    list2 = glob.glob(dir2+"/*.png")
    print("found {} & {} images in two folders".format(len(list1),len(list2)))
    assert len(list1)==len(list2)
    count_diff=0
    for i,j in zip(list1, list2):
        difference = cv2.subtract(cv2.imread(i),cv2.imread(j))
        result = np.any(difference)
        if result:
            cv2.imwrite(DIFF_OUTPUT_DIR + "differece" + os.path.basename(i), difference)
            count_diff += 1
            # print("image {} different, the difference is stored in {}".format(name,DIFF_OUTPUT_DIR))
    print("{} images are different, comparation results saved in {}".format(count_diff, DIFF_OUTPUT_DIR))
    return count_diff

def _combine_dir(basedir):
    dir1=BASEDIR + SQ + "/fusion/maps/semantic_grid_colorized/*.png"
    dir2=BASEDIR + SQ + "/fusion/maps/learning_semantic_grid_sparse_colorized/*.png"
    return dir1,dir2

if __name__ == '__main__':

    BASEDIR="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange5+1/"
    DIFF_OUTPUT_DIR="/tmp/difference_between_images/"
    SQ="04"
    # refresh the output folder:
    if os.path.isdir(DIFF_OUTPUT_DIR):
        shutil.rmtree(DIFF_OUTPUT_DIR)
    os.makedirs(DIFF_OUTPUT_DIR)
    #test1="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_testr5/04/fusion/maps/learning_semantic_grid_dense_colorized"
    #test2="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_testr5/04/fusion/maps/learning_semantic_grid_sparse_colorized"

    test1="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_single/04/fusion/maps/learning_semantic_grid_sparse_colorized"
    test2="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_single/04/fusion/maps/learning_semantic_grid_dense_colorized"
    # checking

    cartesian="/mrtstorage/users/zwang/pcd_mapper_pastonly/cartesian_single/07/grid_map_07/fusion/maps/intensity"
    polar = "/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_single/07/fusion/maps/intensity"
    compare_via_glob(cartesian,polar)
