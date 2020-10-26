import cv2
import numpy as np
import os
import shutil
import glob

def compare_2_image(path1,path2, ifcount=False):
    difference = cv2.subtract(cv2.imread(path1),cv2.imread(path2))
    result = np.any(difference)
    if result:
        cv2.imwrite(DIFF_OUTPUT_DIR + "differece" + os.path.basename(path1), difference)
        print("image {} different, the difference is stored in {}".format(os.path.basename(path1), DIFF_OUTPUT_DIR))
        return 1
    else:
        print("same", end=" ")
        return 0

def compare_folder_via_glob(dir1, dir2):
    list1 = sorted(glob.glob(dir1+"/*.png"))
    list2 = sorted(glob.glob(dir2+"/*.png"))
    len1 = len(list1)
    len2 = len(list2)
    print("found {} & {} images in two folders".format(len1, len2))
    if len1>len2:
        del list1[0:RANGE_DIFF]
    elif len1<len2:
        del list2[0:RANGE_DIFF]
    assert len(list1)==len(list2)

    count_diff=0
    for i,j in zip(list1, list2):
        count_diff +=  compare_2_image(i,j)

    print("{} images are different, comparation results saved in {}".format(count_diff, DIFF_OUTPUT_DIR))

def _combine_dir(basedir):
    BASEDIR="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange5+1/"
    dir1=BASEDIR + SQ + "/fusion/maps/semantic_grid_colorized/*.png"
    dir2=BASEDIR + SQ + "/fusion/maps/learning_semantic_grid_sparse_colorized/*.png"
    return dir1,dir2

if __name__ == '__main__':

    DIFF_OUTPUT_DIR="/tmp/difference_between_images/"
    # refresh the output folder:
    if os.path.isdir(DIFF_OUTPUT_DIR):
        shutil.rmtree(DIFF_OUTPUT_DIR)
    os.makedirs(DIFF_OUTPUT_DIR)




    # for folder comparsion:
    RANGE_DIFF=0
    folder1="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange12_smalltest/04/fusion/maps/learning_semantic_grid_dense_colorized_8class"
    folder2="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange12_smalltest/04/fusion/maps/learning_semantic_grid_dense_colorized"
    #compare_folder_via_glob(folder1, folder2)


    # for single image comarison:
    single1="/home/zwang/Downloads/22.png"
    single2="/home/zwang/Downloads/11.png"
    compare_2_image(single1,single2)
