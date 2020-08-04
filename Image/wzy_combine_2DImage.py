from datasets import Tool_read
import Tool_image
import os.path
import h5py
from PIL import Image


DR_H5_FILE = '/home/zwang/multimodal-deeplab/research/deeplab/datasets/2dposeh5/kitti_07_d.h5'
OLD_XXR_H5_FILE = '/old_xxr_pose_h5/xpose07.h5'
SKITTI_XXR_H5_FILE = '/home/zwang/multimodal-deeplab/research/deeplab/datasets/skitti_xxr_pose_h5/xpose07.h5'
ALL_SEQ = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
IMAGE_SEQ07 = "/home/zwang/Documents/07r1_sparse_colorized/"
IMAGE_TRANSPARENT = '/home/zwang/Documents/07r1_transparent/'
RESOLUTION_X = RESOLUTION_Y = 10  # find it in the parameter file which you used to generate the gridmaps

def _get_dr_single(dr,i):
    return (dr[0,i]*RESOLUTION_X), dr[1,i]

def _get_xyr_single(dr,i):
    """adapt the displacements in meter to gridmaps' pixel based on its resolution"""
    #return int(dr[0,i]*RESOLUTION_X), int(dr[1,i]*RESOLUTION_Y), dr[2,i]
    return int(dr[0,i]*RESOLUTION_X), int(dr[1,i]*RESOLUTION_Y), dr[2,i]


# key frames fro the validation:
#range_list=range(590,591)  #offset -y  # all bad, need x_car offset
range_list=range(100,101)  #offset +x  # all bad, need x_car offset
#range_list=range(750,751)  #vibrate rotation # mine rotation much better
#range_list=range(137,138)   #rotation # mine rotation better



def combine_image_dr(drfilepath,aim_id,imagepath=IMAGE_TRANSPARENT):
    """
    combine past frames, output based on the frame of aim_id
    :param aim_id: aim_id's relative pose stored in dr[aim_id - 1]
    """
    dr = Tool_read.read_pose_h5(drfilepath, paraN=2)
    path_list = Tool_image.read_image_pathlist(imagepath)
    d, r = _get_dr_single(dr, aim_id-1)
    print("relatve d,r =",d,r)
    img1 = Image.open(path_list[aim_id - 1])
    img2 = Image.open(path_list[aim_id])
    img1_r = Tool_image.apply_dr(img1, d, r, ifPast=True)
    output12 = Tool_image.add_images(img1_r, img2)
    output12.show()
    # merge 3 images
    '''
    img3 = Image.open(path_list[i + 2])
    d2, r2 = _get_dr_single(dr, i + 2)
    output12 = wzy_imageTool.apply_dr_past(output12, d2, r2)
    output23 = wzy_imageTool.add_images(output12, img3)
    output23.show()
    '''

def combine_n_image(drfilepath,aim_id, n_frames, imagepath=IMAGE_TRANSPARENT):
    dr = Tool_read.read_pose_h5(drfilepath, paraN=2)
    path_list = Tool_image.read_image_pathlist(imagepath)
    # apply dr to all past frames, add it to the current frame
    img_aim = Image.open(path_list[aim_id])
    d, r = 0, 0
    for n in range(1, n_frames):
        img = Image.open(path_list[aim_id-n])
        d_plus, r_plus = _get_dr_single(dr, aim_id-n)
        #print("relative d,r = ",d_plus,r_plus)
        d = d + d_plus
        r = r + r_plus
        print("absolute d,r = ", d, r, "(by adding up relative)")
        img_r = Tool_image.apply_dr(img, d, r, ifPast=True)
        img_aim = Tool_image.add_images(img_r, img_aim)
    img_aim.show()
    return img_aim

# xyr no longer in use
"""
def combine_image_xyr(xyr_file_path, imagepath=IMAGE_TRANSPARENT): 
    xyr = readTool.read_pose_h5(xyr_file_path, paraN=3)
    path_list = imageTool.read_image_pathlist(imagepath)
    for i in range_list:
        # rotation and offset image i, add it to image i+1
        x,y,r = _get_xyr_single(xyr, i) # 1000 relative pose ~ 1001 images, image i+1's pose stores in i position
        print(x,y,r)
        img1 = Image.open(path_list[i])
        img2 = Image.open(path_list[i + 1])
        img1_r = imageTool.apply_xyr(img1, x, y, r, ifPast=True)
        output12 = imageTool.add_images(img1_r, img2)
        output12.show()
"""


if __name__ == '__main__':
    """
    key frames to check in seq 07
    590: offset -y
    100: offset +x
    750: vibrate rotation
    137: rotation
    """

    key_frames_to_check=900
    #combine_image_dr(DR_H5_FILE,f_to_check)
    #combine_image_dr(OLD_XXR_H5_FILE,f_to_check)
    #combine_image_dr(SKITTI_XXR_H5_FILE,f_to_check)

    #combine_n_image(DR_H5_FILE,137,2)
    #combine_n_image(SKITTI_XXR_H5_FILE,f_to_check,10)
    combine_n_image(SKITTI_XXR_H5_FILE, key_frames_to_check, 1)
    combine_n_image(SKITTI_XXR_H5_FILE, key_frames_to_check, 10)
