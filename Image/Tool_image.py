from scipy.spatial.transform import Rotation as R
import os.path
import h5py
from PIL import Image
from PIL import ImageChops
import glob
import numpy as np
import matplotlib.pyplot as plt

#PATH_MATRIX_R='C:/Users/Administrator/OneDrive/workspace/02relativeM.txt'
ALL_SEQ = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
IMAGE_SEQ07 = "/home/zwang/Documents/07r1_sparse_colorized"
IMAGE_TRANSPARENT = '/home/zwang/Documents/07r1_transparent/'


def read_image_pathlist(filepath):
    image_path_list = sorted(glob.glob(os.path.join(filepath,"*.png")))
    #print(len(image_path_list)," images are found")
    return image_path_list


def _transparent(image_path,out_path=IMAGE_TRANSPARENT):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    name= os.path.basename(image_path)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    img.save(str(out_path+name), "PNG")


def transparent_root(image_root_path):
    image_path_list = sorted(glob.glob(os.path.join(image_root_path,"*.png")))
    #print(len(image_path_list), " images are found")
    for image_path in image_path_list:
        _transparent(image_path)
    print("done")


def add_images(img1,img2):
    """
       based on image2
    """
    datas1 = img1.getdata()
    datas2 = img2.getdata()
    newdata=[]
    for item1,item2 in zip(datas1,datas2):
        # only add image if the pixel in image 2 is empty
        if item2[3]==0:
            newdata.append(item1)
        else:
            newdata.append(item2)
    img1.putdata(newdata)
    return img1


def add_images_path(imagepath1, imagepath2):
    img1 = Image.open(imagepath1)
    img2 = Image.open(imagepath2)
    return add_images(img1,img2)


def apply_dr(img,d,r,ifPast=True):
    """
    inverse the distance and rotation,
    apply to the past image, align it to the current
    """
    if ifPast:
        img = img.rotate(-r)
        img = ImageChops.offset(img, xoffset=int(-(d+0.5)), yoffset=0)
        return img
    else:
        raise RuntimeError("not worte yet @wzy")


def apply_xyr(img, x,y,r,ifPast=True):
    """rotate, then offset"""
    if ifPast:
        img = img.rotate(-r)
        img = ImageChops.offset(img, xoffset=int(-(d+0.5)), yoffset=int(-y))
        return img
    else:
        img = img.rotate(r)
        img = ImageChops.offset(img, xoffset=int(-(d+0.5)), yoffset=int(y))
        return img
