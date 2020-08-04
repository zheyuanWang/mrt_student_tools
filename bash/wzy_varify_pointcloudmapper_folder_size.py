import glob
import os
import cv2

# count the items in the folders of the pointcloud_mapper's output
# count the 6 layers and 2 semantic folders
# cross compare the Nr of items(images) of different BASEs
BASE1="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_single/"
BASE5="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange5+1/"
BASE10="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange10/"
BASE20="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_pastrange20/"
SQ_ALL = ('00','01','02','03','04','05','06','07','08','09','10')

#HELP
#how to produce sparse label?  use semantic-kitti-apiremapLabelsOfGridMap.py


#some command to clean up the mess:

#delete the files older than 48h
# find ./  -mmin +2880 -delete

#remove first 18 images
#rm */000001.png */000002.png */000003.png */000004.png */000005.png */000006.png */000007.png */000008.png */000009.png */000010.png */000011.png */000012.png */000013.png */000014.png */000015.png */000016.png */000017.png */000018.png

LIST_FOLDERS_6=["/fusion/maps/detections/*.png", "/fusion/maps/detections_z_min/*.png","/fusion/maps/detections_z_max/*.png", "/fusion/maps/intensity/*.png", "/fusion/maps/observations/*.png", "/fusion/maps/occlusions_z_upper_bound/*.png"]
LIST_FOLDERS_semantic=["/fusion/maps/semantic_grid/*.png", "/fusion/maps/learning_semantic_grid_sparse/*.png", "/fusion/maps/learning_semantic_grid_dense/*.png", "/fusion/maps/semantic_grid_colorized/*.png","/fusion/maps/learning_semantic_grid_sparse_colorized/*.png", "/fusion/maps/learning_semantic_grid_dense_colorized/*.png"]
LIST_FOLDERS_all = LIST_FOLDERS_6 + LIST_FOLDERS_semantic
def compare_size(basedir, sequence=SQ_ALL):
    print("\nchecking size in {}".format(basedir))
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

        if(detection==det_z_min==det_z_min==intensity==observations==observ_z_min!=0):
            #print("checking {}: ok 6-layers with size={}".format(seq,detection))
            returnlist.append(detection)
            if not (detection==semanticg==semanticcolor):
                print("             semantic_GRID foldersize ={},{} in sequence{}".format(semanticg,semanticcolor,seq))
            if not (se_sparseg==se_sparseg_color==se_denseg==se_denseg_color==detection):
                print("             SPARSE semantic foldersize ={},{},{},{} in sequence{}".format(se_sparseg,
                                                                                                  se_sparseg_color,
                                                                                                  se_denseg,
                                                                                                  se_denseg_color, seq))
        else:
            checklist = [detection, det_z_min, det_z_max, intensity, observations, observ_z_min]
            print("size of 6 channel unequal/ =0",basedir,"\n",seq,checklist)
    print("done")
    checkTFrecords(basedir)
    return returnlist

def checkTFrecords(basedir):

    if os.path.isdir(basedir+"train"):
        print("The last TFrecord is")
        print(max(glob.glob(basedir+"train/*"), key = os.path.getctime)[-44:])
    else:
        print("no TF in train")
    if os.path.isdir(basedir+"val"):
        print(max(glob.glob(basedir+"val/*"), key = os.path.getctime)[-44:])
    else:
        print("no TF in val")


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
    CHECKNr=False
    CHECK_IMAGE_SIZE=True
    if(CHECKNr):
        blist1=compare_size(BASE1)
        blist5=compare_size(BASE5)
        blist10=compare_size(BASE10)
        blist20=compare_size(BASE20)
        print(
            "\nthe size of all sequence\n{}\n{}\n{}\n[0000, 0001, 0002, 003, 004, 0005, 0006, 0007, 0008, 0009, 0010]".format(
                blist5, blist10, blist20))
        print("\ncross compare: diff should be same in each list")
        cross_compare2(blist5,blist10)
        cross_compare2(blist10,blist20)
        cross_compare2(blist20,blist5)
        cross_compare2(blist1,blist20)
    if (CHECK_IMAGE_SIZE):
        check_image_size(BASE1,listfolders=LIST_FOLDERS_semantic)


