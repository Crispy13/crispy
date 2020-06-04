import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
import cv2 # For processing images
import matplotlib.pyplot as plt
from matplotlib import colors
import math
from copy import deepcopy, copy
import pandas as pd
from ecf import *
import random
from itertools import product
from operator import itemgetter
import itertools
from scipy.ndimage import rotate
import sys

from .eclogging import load_logger
logger = load_logger()

def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def read_nii(img_seg_dict, types=('t1ce', 'seg')):
    result_dict={}

    for i, v in img_seg_dict.items():
        result_dict[i]=sitk.GetArrayFromImage(sitk.ReadImage(v))
    
    return result_dict

def resize(img, shape, mode='constant', orig_shape=None, order=3):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    
    if orig_shape == None: orig_shape = img.shape
    
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1], 
        shape[2]/orig_shape[2]
    )
    
    # Resize to the given shape
    return zoom(img, factors, mode=mode, order=order)


def preprocess(img, out_shape=None, orig_shape=None, normalization=True, only_nonzero_element=True):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant', orig_shape=img.shape)
    
    # Normalize the image (only each element is not zero.)
    if normalization == False: return img
    
    if only_nonzero_element == True:
        p=np.where(img!=0)
        mean=img[p].mean()
        std=img[p].std()

        result_array=np.where(img!=0, (img-mean)/std, img)
    else:
        mean=img.mean()
        std=img.std()
        result_array = (img - mean) / std

    return result_array


def preprocess_label(img, out_shape=None, mode='nearest', closing=False):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)
    
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)
    
    if closing == True:
        kernel = np.ones((3, 3))
        
        for t in [ncr, ed, et]:
            for z in range(len(t)):
                t[z]=cv2.morphologyEx(t[z], cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return np.array([ncr, ed, et], dtype=np.uint8)

def preprocess_label_(img, out_shape=None, mode='nearest', label='all', closing=False, zoom_order=3):
    """
    The sub-regions considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC/1+4), and 3) the "whole tumor" (WT/1+2+4) 
    
    label : 'all' or list of label number. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    
    # Select labels.
    if label == 'all':
        img = np.where(img > 0, 1, img)   
    elif len(label) == 2:
        img = np.where((img == label[0]) | (img == label[1]), 1, 0)
    elif len(label) == 1:
        img = np.where(img == label, 1, 0)
    elif label == 'Brats':
        et = (img == 4)  # enhancing tumor
        tc = (img == 1) | (img==4) # tumor core
        wt = (img == 1) | (img==2) | (img==4) # whole tumor
    
        if out_shape is not None:
            et = resize(et, out_shape, mode=mode, order=zoom_order)
            tc = resize(tc, out_shape, mode=mode, order=zoom_order)
            wt = resize(wt, out_shape, mode=mode, order=zoom_order)
        
        return np.array([et, tc, wt], dtype=np.uint8)
    
    else:
        raise Exception("Label argument is not valid.")
    
    if out_shape is not None:
        img = resize(img, out_shape, mode=mode, order=zoom_order)
    
    return np.array([img], dtype = np.uint8)


def prepare_data(data_w_num, resize_output_shape = None,
                 only_nonzero=False, label_criteria = None, label_zoom_order = 0, img_types = None):
    i, imgs = data_w_num

    try:
        d = np.array(
            [preprocess(imgs[m], resize_output_shape, only_nonzero_element=only_nonzero) for m in img_types[:-1]],
            dtype=np.float32)
        l = preprocess_label_(imgs['seg'], resize_output_shape, zoom_order=label_zoom_order, label=label_criteria)
        
        # Print the progress bar
#         increment()
#         print(f'\r{counter.value}/{total} has been completed.', end='')
        return i, d, l
        
    except Exception as e:
        print(f'Something went wrong with {i}th file, skipping...\n Exception:\n{str(e)}')
        return i, str(e), str(e)
    
    
def find_found_path(target, search_string='Nothing'):
    l=list(map(lambda x:re.search(search_string, x)==None, target))

    indices = [i for i, v in enumerate(l) if v==True]

    return indices

def search_file_w_kw(target, keyword, path_pattern='(.+)/.*?pre/.*?$'):
    """
    keyword : a list of keywords.
    path_pattern = a regex pattern which has a group of path in which you want to search files.
    """
    
    r=[]
    cl=[]
    k=f"(?:{str.join('|', keyword)}).*\.nii\.gz"
    
    
    for c, i in enumerate(target):
        re_r1 = re.search(path_pattern, i).group(1)
#         print(re_r1)
        gr1 = glob.glob(f"{re_r1}/*")
#         print(gr1)
        ir1 = list(filter(lambda x:re.search(k, x), gr1))
        if len(ir1) == 0:
            ir = [f'Nothing was found. path:{re_r1}']
        else:
            if len(ir1) != 1: cl.append([c, ir1])
            ir=ir1
        r.append(ir)
    
#     r=list(itertools.chain(*r))
    
    return r, cl

def crop_image_(img, crop_size=None, mode='center'):
    assert crop_size != None, "Crop size should be passed."
    print(img.shape)
    c, h, w, d=img.shape
    cc, ch, cw, cd=crop_size
    
#   print(h,w,d,'\n',ch,cw,cd)
    cropped_image=np.empty(shape=crop_size)
    for i in range(len(cropped_image)):
        cropped_image[i]=img[i][h//2 - ch//2 : h//2 + ch//2, w//2 - cw//2 : w//2 + cw//2, d//2 - cd//2 : d//2 + cd//2]
    
    return cropped_image

def output_even(x):
    if x % 2 == 0:
        return x
    else:
        return x + 1

def auto_crop(data_and_label, mode=None, buffer_size=10, debug=False):
    """
    return cropped [img, label]
    
    data_and_label : list of 3d-array numpy image. e.g. [data, labels]
    
    crop area = (x of estimated brain area + 2 * buffer_size) * (y of estimated brain area + 2 * buffer_size)
    """
    imgs = data_and_label[:-1]
    label = data_and_label[-1]

    rl = [] # ranges_list
    for img in imgs:
        p = np.where(img != img.min())

        z_range=[p[0].min(), p[0].max()]
        y_range=[p[1].min(), p[1].max()]
        x_range=[p[2].min(), p[2].max()]

        cz=(z_range[1] + z_range[0]) // 2
        cy=(y_range[1] + y_range[0]) // 2
        cx=(x_range[1] + x_range[0]) // 2

        rz=z_range[1] - z_range[0]
        ry=y_range[1] - y_range[0]
        rx=x_range[1] - x_range[0]

        bs=buffer_size

        z_range = [i if i>=0 else 0 for i in [z_range[0] - bs, z_range[1] + bs]]
        y_range = [i if i>=0 else 0 for i in [y_range[0] - bs, y_range[1] + bs]]
        x_range = [i if i>=0 else 0 for i in [x_range[0] - bs, x_range[1] + bs]]
        
        rl.append([z_range, y_range, x_range])
    
    if rl.count(rl[0]) == len(rl):
        z_range, y_range, x_range = rl[0] ; logger.debug(f"ranges are same.")
    else:
        z_range, y_range, x_range = list(zip([min([r[i][0] for r in rl]) for i in range(3)],
                                             [max([r[i][1] for r in rl]) for i in range(3)])) ; logger.debug(f"ranges are different.")
    
    if debug:
        print('z_range: ', z_range, 'y_range: ' , y_range, 'x_range: ', x_range)
    
    r_imgs = [img[z_range[0] : z_range[1], y_range[0] : y_range[1], x_range[0] : x_range[1]] for img in imgs]
    label = label[z_range[0] : z_range[1], y_range[0] : y_range[1], x_range[0] : x_range[1]]
    
    return [*r_imgs, label], [z_range, y_range, x_range]

def crop_image(img, crop_size=None, mode='center'):
    assert crop_size != None, "Crop size should be passed."
    print(img.shape)
    c, h, w, d=img.shape
    cc, ch, cw, cd=crop_size

#   find the range of coordinates of brain

#   print(h,w,d,'\n',ch,cw,cd)
    cropped_image=np.empty(shape=crop_size)
    for i in range(len(cropped_image)):
        cropped_image[i]=img[i][h//2 - ch//2 : h//2 + ch//2, w//2 - cw//2 : w//2 + cw//2, d//2 - cd//2 : d//2 + cd//2]
    
    return cropped_image


