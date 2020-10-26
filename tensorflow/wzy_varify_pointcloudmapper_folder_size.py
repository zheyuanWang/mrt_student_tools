import glob
import os
import cv2
from termcolor import colored


# count the items in the folders of the pointcloud_mapper's output
# count the 6 layers and 2 semantic folders
# cross compare the Nr of items(images) of different BASEs
BASE1="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_single/"
BASE4="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange4/"
BASE5="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange5+1/"
BASE8="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange8/"
BASE10="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/"
BASE12="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange12/"
BASE15="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange15/"
BASE16="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange16/"
BASE20="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange20/"
BASE24="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange24/"
INT4="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval4/"
INT6="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval6/"
INT16="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval16/"
INT20="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval20/"
INT24="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_baseline_2F_interval24/"
MID12="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_middle12/"
MID20="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_middle20/"

SQ_ALL = ('00','01','02','03','04','05','06','07','08','09','10')

"""
how to produce sparse label?  
    use semantic-kitti-apiremapLabelsOfGridMap.py

some command to clean up the mess:

    delete the files older than 48h
    find ./  -mmin +2880 -delete

remove first 18 images
    rm */000001.png */000002.png */000003.png */000004.png */000005.png */000006.png */000007.png */000008.png */000009.png */000010.png */000011.png */000012.png */000013.png */000014.png */000015.png */000016.png */000017.png */000018.png
"""

def cprint(text, c=3):
    color_to_choose = ['red' ,'yellow', 'green', 'cyan', 'blue', 'white', 'magenta', 'grey']
    # Nr of color        0        1       2         3       4        5        6        7
    if c > 7:
        c = c % 7
    color = color_to_choose[c]
    print(colored(text, color))
    return


LIST_FOLDERS_6=["/fusion/maps/detections/*.png", "/fusion/maps/detections_z_min/*.png","/fusion/maps/detections_z_max/*.png", "/fusion/maps/intensity/*.png", "/fusion/maps/observations/*.png", "/fusion/maps/occlusions_z_upper_bound/*.png"]
LIST_FOLDERS_semantic=["/fusion/maps/semantic_grid/*.png", "/fusion/maps/learning_semantic_grid_sparse/*.png", "/fusion/maps/learning_semantic_grid_dense/*.png", "/fusion/maps/semantic_grid_colorized/*.png","/fusion/maps/learning_semantic_grid_sparse_colorized/*.png", "/fusion/maps/learning_semantic_grid_dense_colorized/*.png"]
LIST_FOLDERS_all = LIST_FOLDERS_6 + LIST_FOLDERS_semantic


def checkTFrecords(basedir):

    if os.path.isdir(basedir+"train"):
        print("    The last TFrecord is")
        print(max(glob.glob(basedir+"train/*"), key=os.path.getmtime)[-44:])
    else:
        if len(glob.glob(basedir+"/*.tfrecord")) != 0:
            cprint("    detected TF in sequence root dir, need to move",2)
        else:
            print("    no TF in train")
    if os.path.isdir(basedir+"val"):
        print(max(glob.glob(basedir+"val/*"), key=os.path.getctime)[-44:])
    else:
        print("    no TF in val")


def compare_size(basedir, sequence=SQ_ALL):
    cprint("\nchecking size in {}".format(basedir))
    returnlist=[]
    for seq in sequence:
        detection = len(glob.glob(basedir + seq + "/fusion/maps/detections/*.png"))
        det_z_min = len(glob.glob(basedir + seq + "/fusion/maps/detections_z_min/*.png"))
        det_z_max = len(glob.glob(basedir + seq + "/fusion/maps/detections_z_max/*.png"))
        intensity = len(glob.glob(basedir + seq + "/fusion/maps/intensity/*.png"))
        observations = len(glob.glob(basedir + seq + "/fusion/maps/observations/*.png"))
        observ_z_min = len(glob.glob(basedir + seq + "/fusion/maps/occlusions_z_upper_bound/*.png"))

        semanticg = len(glob.glob(basedir + seq + "/fusion/maps/semantic_grid/*.png"))
        semanticcolor = len(glob.glob(basedir + seq + "/fusion/maps/semantic_grid_colorized/*.png"))
        se_sparseg = len(glob.glob(basedir + seq + "/fusion/maps/learning_semantic_grid_sparse/*.png"))
        se_sparseg_color = len(glob.glob(basedir + seq + "/fusion/maps/learning_semantic_grid_sparse_colorized/*.png"))
        se_denseg = len(glob.glob(basedir + seq + "/fusion/maps/learning_semantic_grid_dense/*.png"))
        se_denseg_color = len(glob.glob(basedir + seq + "/fusion/maps/learning_semantic_grid_dense_colorized/*.png"))

        # for semantic_full:
        semantic_full = len(glob.glob(basedir + seq + "/fusion/maps/semantic_full/*_channel_15.png"))
        if semantic_full != detection:
            cprint("    seq{}: semantic_full has {} images != {}".format(seq, semantic_full, detection),1)

        if(detection==det_z_min==det_z_min==intensity==observations==observ_z_min!=0):
            #print("checking {}: ok 6-layers with size={}".format(seq,detection))
            returnlist.append(detection)
            if not (detection==semanticg==semanticcolor):
                print("        seq{}: semantic_GRID foldersize ={},{} != {}".format(seq, semanticg, semanticcolor,detection))
            if not (se_sparseg==se_sparseg_color==se_denseg==se_denseg_color==detection):
                print("        seq{}: 4x learning_XXXX foldersize ={},{},{},{}".format(seq, se_sparseg,
                                                                                                  se_sparseg_color,
                                                                                                  se_denseg,
                                                                                                  se_denseg_color))
        else:
            checklist = [detection, det_z_min, det_z_max, intensity, observations, observ_z_min]
            cprint("    seq{}: size of 6 channels != {} or = 0 : \n        list= {}".format(seq,detection,checklist), 0)
            cprint("        tip: have you moved the content of grid_map_xx?",6)
    cprint("    -- passed varifacation if no info above --",4)
    checkTFrecords(basedir)
    return returnlist


def cross_compare2(list1,list2):
    if (len(list1)==len(list2)):
        Nrseq=len(list1)
    else:
        raise RuntimeError("length of liset unequal")
    returnlist=[]
    for s in range(Nrseq):
        returnlist.append(list1[s]-list2[s])
    print(returnlist)

def check_image_size(basedir,listfolders,sequence=SQ_ALL):
    for seq in sequence:
        print("checking image size in seq", seq)
        for folder_fname in listfolders:
            path2image_list = glob.glob(basedir + seq + folder_fname)
            im0 = cv2.imread(path2image_list[0])
            for path2image in path2image_list:
                im = cv2.imread(path2image)
                if not im.shape == im0.shape:
                    print("Diff in image size!, the size of {}is {}".format(path2image,im.shape))
            print("done with image size in", folder_fname, "size=",im0.shape)


if __name__ == '__main__':
    CHECKNr=True
    if(CHECKNr):
        #listname4 = compare_size(BASE24)
        #listname3 = compare_size(BASE16)
        listname2 = compare_size(BASE16)
        listname1 = compare_size(BASE24)
        #listname5 = compare_size(BASE4)
        #listname6 = compare_size(MID12)

        print(
            "\nthe size of all sequence\n{}\n{}[0000, 0001, 0002, 003, 004, 0005, 0006, 0007, 0008, 0009, 0010]".format(
                listname1, listname2))
        print("\ncross compare: diff should be same in each list")

        cross_compare2(listname1,listname2)
        #cross_compare2(listname1,listname3)
        #cross_compare2(listname1,listname4)
        #cross_compare2(listname1,listname5)
        #cross_compare2(listname1,listname6)


    CHECK_IMAGE_SIZE = False
    if (CHECK_IMAGE_SIZE):
        check_image_size(BASE1,listfolders=LIST_FOLDERS_semantic)