import numpy as np
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
import random
from itertools import product
from operator import itemgetter
import itertools
from scipy.ndimage import rotate
import sys
from itertools import chain
import os, traceback
import tensorflow as tf
from tensorflow import keras
from .eckeras import *
from .ecf import *
import _pickle as pickle

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
    
    Parameters
    ----------
    img : a numpy array, e.g. (H,W,D) , (H,W)
    shape : a target shape
    
    """
    
    if orig_shape == None: orig_shape = img.shape
    
    factors = [shape[i]/orig_shape[i] for i in range(len(img.shape))]
    
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
    
    if (out_shape is not None) and (label != 'Brats'):
        img = resize(img, out_shape, mode=mode, order=zoom_order)
#         img = np.clip(img, 0, 1)
#         img = np.where(img >= 0.5, 1, 0)
    
    return np.array([img], dtype = np.uint8)

def prepare_data(data_w_num, resize_output_shape = None,
                 only_nonzero=False, label_criteria = None, label_zoom_order = 0, label_zoom_mode = 'nearest', img_types = None):
    i, imgs = data_w_num

    try:
        d = np.array(
            [preprocess(imgs[m], resize_output_shape, only_nonzero_element=only_nonzero) for m in img_types[:-1]],
            dtype=np.float32)
        l = preprocess_label_(imgs['seg'], resize_output_shape, mode = label_zoom_mode, zoom_order=label_zoom_order, label=label_criteria)
        
        # Print the progress bar
#         increment()
#         print(f'\r{counter.value}/{total} has been completed.', end='')
        return i, d, l
        
    except Exception as e:
        print(f'Something went wrong with {i}th file, skipping...\n')
        traceback.print_exc()
        return i, str(e), str(e)
    

def find_found_path(target, search_string='Nothing'):
    l=list(map(lambda x:re.search(search_string, x)==None, target))

    indices = [i for i, v in enumerate(l) if v==True]

    return indices

def search_file_w_kw(target, keyword, path_pattern='(.+)/.*?pre/.*?$'):
    """
    keyword : a list of keywords.
    path_pattern = a regex pattern which has a group of path in which you want to search files.
    
    Examples
    --------
    In the following situation:
        a t1ce file path: /data/eck/Workspace/snubh/SNUH-bias/part1/10026333/2012-07-20/10026333_2012-07-20_pre/T1_regi_brain_bias_restore.nii.gz
        the path of seg file of the t1ce: /data/eck/Workspace/snubh/SNUH-bias/part1/10026333/2012-07-20/10026333_2012-07-20_pre/seg_regi.nii.gz
        keyword: ['seg', 'roi', 'label']
        
    You can use this function like:
        search_file_w_kw(t1ce, keyword=['roi', 'seg', 'label'], path_pattern='(.+/.*?pre)/.*?$')
            
    
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

def auto_crop(data_and_label, mode=None, buffer_size=10, thresh_percent = 0.1, debug=False):
    """
    return cropped [img, label]
    The threshold is calculated by:
        min + (max - min) * thresh_percent
    
    If thresh_percent is 0, it means the threshold = min intensity of the image.
    
    Parameters
    ----------
    data_and_label : list of 3d-array numpy image. e.g. [data, labels]
    thresh_percent :
    
    """
    imgs = data_and_label[:-1]
    label = data_and_label[-1]

    rl = [] # ranges_list
    
    for img in imgs:
        threshold = img.min() + (img.max()-img.min()) * thresh_percent
        p = np.where(img > threshold)

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


###
def auto_crop_using_mask(data_and_label, mask, mode=None, buffer_size=10, thresh_percent = 0.1, debug=False):
    """
    return cropped [img, label]
    The threshold is calculated by:
        min + (max - min) * thresh_percent
    
    If thresh_percent is 0, it means the threshold = min intensity of the image.
    
    Parameters
    ----------
    data_and_label : list of 3d-array numpy image. e.g. [data, labels]
    thresh_percent :
    
    """
    imgs = data_and_label[:-1]
    label = data_and_label[-1]

    p = np.where(mask > mask.min())

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


def show_dnl(data, label, sample_index, slice_index):
    Nr=1
    Nc=2
    
    figs, axes = plt.subplots(Nr, Nc, figsize=[10,8])
    figs.suptitle('data and label')
    
    axes[0].imshow(data[sample_index][0][slice_index], cmap='Greys_r')
    axes[1].imshow(label[sample_index][0][slice_index], cmap='Greys_r')
#     for i in range(Nr):
#         for j in range(Nc):
#             images.append(axs[i,j].imshow(data, cmap='Greys_r'))
#             initial_index += 1

def dice_numpy(y_true, y_pred, avg_over_samples = True, e=1e-8):
    assert avg_over_samples in [True, False]
    
    intersection = np.sum(np.abs(y_true * y_pred), axis = (-3,-2,-1))
    dn = np.sum(np.square(y_true) + np.square(y_pred), axis = (-3, -2, -1)) + e
    if avg_over_samples is True:
        return np.mean(2 * intersection / dn, axis = (0,1))
    else:
        return np.mean(2 * intersection / dn, axis = 1)
        

### Data Augmentation Functions
def flip_(img, label, axis=None):
    """
    img : 4d-array expected.
    """
    
    assert img.ndim == 4
    if axis == 1:
        r = (img[:, ::-1 , ...], label[:, ::-1, ...])
    elif axis == 2:
        r = (img[..., ::-1 , :], label[..., ::-1 , :])
    elif axis == 3:
        r = (img[..., ::-1], label[..., ::-1])
    
    return r
    

def flip(img, label, flip_mode='all', **kwargs):
    """
    This function returns a tuple of tuples each of which includes flipped (img, label).
    
    """
    # A Random Probability 0.5 is not applied to this function.

    result = []
    
    choose = list(product([False, True], [False, True], [False, True]))
    choose.remove((False, False, False))
    axes=np.array([1,2,3])
    
    if flip_mode == 'all':
        for i in choose:
            i=list(i)
            axes_tbu = axes[i]
            ir = img, label
            for j in axes_tbu:
                ir=flip_(ir[0], ir[1], axis=j)
            result.append(ir)
        
        return np.array(result)
    
    elif flip_mode == 'prob':
        probs = list(itertools.product([False, True], [False, True], [False, True]))
        probs.remove((False, False, False))
        prob = list(random.choice(probs))
        
        for i in axes:
            result.append(flip_(img, label, axis=i))
            
        return np.array(result)[prob]
            
    elif flip_mode == 'xyz':
        prob = [True, True, True]
        
        for i in axes:
            result.append(flip_(img, label, axis=i))
            
        return np.array(result)[prob]
    
def scale(img, label, **kwargs):
    """
    This function returns a tuple of 2 tuples each of which has scaled or shifted image and a label.
    
    img : 4d-array image
    label : 4d-array label
    """
    assert img.ndim == 4
    assert label.ndim == 4
    
    c = img.shape[0] # channels
    
    scale=np.random.uniform(0.9, 1.1, c).reshape(c, 1, 1, 1)
    
    scaled = img * scale
    
    return [[scaled, label]]

def shift(img, label, **kwargs):
    """
    This function returns a tuple of 2 tuples each of which has scaled or shifted image and a label.
    
    img : 4d-array image
    label : 4d-array label
    """
    assert img.ndim == 4
    assert label.ndim == 4
    
    c = img.shape[0] # channels
    
    shift=np.random.uniform(-0.1, 0.1, c).reshape(c, 1, 1, 1)
    
    shifted = img + shift
    
    return [[shifted, label]]

def rescale(x, max_value = 255, axis = (-3, -2, -1)):
    """
    Min Max scale.
    """
    ndim = x.ndim
    if ndim not in [3, 4]: logger.debug("func rescale: Data for rescale hasn't length of 4 or 3.")
    
    def make_max_min(x, mode, axis = axis):
        assert mode in ['max', 'min']
        d = x.max(axis = axis) if mode == 'max' else x.min(axis = axis)
        
        if ndim == 4:
            c = x.shape[0]
            d = d.reshape(c, 1, 1, 1)
            
        return d

    # Main process
    min_arr = make_max_min(x, 'min')
    
    if x.min() < 0:
        x = x - min_arr
        min_arr = make_max_min(x, 'min')
        
    max_arr = make_max_min(x, 'max')

    x = max_value * ((x - min_arr) / (max_arr - min_arr))
    
    return x


def std(x):
    """
    Standardization
    """
    
    if len(x.shape) not in [3, 4]: print("Data for standardization hasn't length of 4 or 3.", file=sys.stderr)
    
    if len(x.shape) == 4:
        axis = (-3, -2, -1)
        mean_arr = x.mean(axis = axis)[None, None, None, ...]
        std_arr = x.std(axis = axis)[None, None, None, ...]
    else:
        axis = None
        mean_arr = x.mean(axis = axis)
        std_arr = x.std(axis = axis)
        
        
    x = (x-mean_arr) / std_arr
    
    return x


def change_brightness(data, label, br_values = [-0.5, 0.5], **kwargs):
    assert len(data.shape) == 4, "Data's length should be 4."
    
    data = deepcopy(data)
    data = rescale(data, 255)
    
    r = []
    
    for v in br_values:
        ir = []
        for d in data:
            ir.append(std(np.clip(d + v * 255, 0, 255)))
        
        r.append([np.stack(ir), label])
    
    return r

def change_contrast(data, label, factors = [float(1e-4), 1.5], **kwargs):
    assert len(data.shape) == 4, "Data's length should be 4."
    
    data = rescale(deepcopy(data), 255)
    
    r = []
    for factor in factors:
        factor = float(factor)
        ir = []
        for d in data:
            ir.append(std(np.clip(128 + factor * d - factor * 128, 0, 255)))
        
        r.append([np.stack(ir), label])
        
    return r

def jittering(data, label, ji_ms = [-25, 9], only_seg = False, **kwargs):
    assert len(data.shape) == 4, "Data's length should be 4."
    assert only_seg in [True, False], "only_seg argument should be True or False"
    
    data = rescale(deepcopy(data), 255)
    m, s = ji_ms[0], ji_ms[1]
    
    ir = []
    seg_pos = np.where(label[0] == 1)
    for d in data:
        noise = np.random.normal(m, s, d.shape)
        if only_seg is False:
            ir.append(std(np.clip(d + noise, 0, 255)))
        else:
            d[seg_pos] = d[seg_pos] + noise[seg_pos]
            ir.append(std(np.clip(d, 0, 255)))
    
    return [[np.stack(ir), label]]
    
def downsize_and_padding(data, label):
    return

def random_rotation(data, label, rr_reshape = False, rr_rotate_order = 3, rr_rotate_label_order = 3, rr_label_mode = 'nearest', rr_resize_label_order = 0, **kwargs):
    data = rescale(data, 255)
    original_shape = data.shape[1:]
    
    # Set angle and axis
    angle = np.random.uniform(30, 330) ; axis = random.choice([(0, 1), (0, 2), (1, 2)])
    
    ir = []
    for d in data:
        d = rotate(d, angle, axis, reshape = rr_reshape, order = rr_rotate_order, cval = d[0,0,0])
        if rr_reshape:
            d = resize(d, shape = original_shape)
        ir.append(std(d))
    
    ir2 = []
    for l in label:
        l = rotate(l, angle, axis, mode = rr_label_mode, reshape = rr_reshape, order = rr_rotate_label_order, cval = 0)
        if rr_reshape:
            l = resize(l, shape = original_shape, mode = rr_label_mode, order = rr_resize_label_order)
        ir2.append(l)
    
    return [[np.stack(ir), np.stack(ir2)]]

def blurring(data, label):
    return


def augmentation_pipeline(data, label, logged, aug_proc = [], aug_mode = 'gradual', additional_samples = 10, **kwargs):
    """
    flip_mode : 'all' or 'prob'
    proc_ref = dict(br = change_brightness, co = change_contrast, ji = jittering, scale = scale, shift = shift, fl = flip, rr=random_rotation)
    
    """
    result=[]
    
    # Check aug_proc's element belong to the reference list.
    proc_ref = dict(br = change_brightness, co = change_contrast, ji = jittering, scale = scale, shift = shift, fl = flip, rr=random_rotation)
    if aug_proc == 'all': aug_proc = list(proc_ref.keys())
    assert set(aug_proc).issubset(list(proc_ref.keys())), f"The elements of aug_proc argument should be subset of reference. {list(proc_ref.keys())}"
    assert set([aug_mode]).issubset(['random', 'gradual']), f"Invalid aug_mode argument:{aug_mode}. referece : ['random', 'gradual']"

    ir = []
    t_logs = []
    if aug_mode == 'random':
        
        for i in range(additional_samples):
            d = deepcopy(data)
            l = deepcopy(label)
            procs = np.random.choice(aug_proc, size = np.random.randint(1, len(aug_proc)+1), replace = False, p = None).tolist()
            for k in ['scale', 'shift']:
                if k in procs:
                    procs.remove(k)
                    procs = procs + [k]
            
            logger.debug(f"selected procedures : {procs}")
            
            for m in procs:
                ar = proc_ref[m](d, l, **kwargs)
                if len(ar) == 0:
                    continue
                elif len(ar) == 1:
                    d, l = ar[0]
                elif len(ar) > 1:
                    d, l = ar[np.random.randint(0, len(ar))]
                
            t_log = deepcopy(logged)
            t_log['augmentation'] = procs
            t_logs.append(t_log)   
            
            ir.append([[d,l]])
            
    elif aug_mode == 'gradual':
        for m in aug_proc:
            ire = proc_ref[m](data, label, **kwargs) ; logger.debug(f"m, len(ire): {m}, {len(ire)}")
            ir.append(ire)
            
            t_log = deepcopy(logged)
            t_log['augmentation'] = m
            t_logs.append([t_log] * len(ire))
        
    r= np.concatenate(ir, axis=0)
    
    ad=[]; al=[]
    
    for i in r:
        ad.append(i[0])
        al.append(i[1])
    
    ad=np.array(ad)
    al=np.array(al)
    
    return [ad, al, list(chain(*t_logs))]


# Run data augmentation pipeline
def run_aug_pipeline(data, label, logged, aug_proc, aug_mode, aug_pipe_args, seed_number = 42, data_dtype = np.float32, **kwargs):
    assert (data.ndim == 5) & (label.ndim == 5), "the ndims of data and label should be 5."
    total = data.shape[0]

    random.seed(seed_number)
    np.random.seed(seed_number)

    ad=[]
    al=[]
    alo = []
    for i, d, l in zip(range(total), data, label):
        rd, rl, rlo = augmentation_pipeline(d, l, logged[i], aug_proc = aug_proc, aug_mode = aug_mode,
                                       **aug_pipe_args)

        ad.append(rd) ; logger.debug(f"len(rd): {len(rd)}")
        al.append(rl) ; logger.debug(f"len(rl): {len(rl)}")
        alo = alo + rlo ; logger.debug(f"len(rlo): {len(rlo)}")
        
        # Print the progress bar
        print_progressbar(total, i)
    
    ad=np.vstack(ad).astype(data_dtype) ; logger.info(f"{ad.shape[0]} images were generated.")
    al=np.vstack(al).astype(np.uint8)
    
    return ad, al, alo


def data_augmentation(x_train, y_train, l_train, aug_proc, aug_mode, aug_pipe_args, save_dir, seed_number = 42, data_dtype = np.float32, **kwargs):
    ad, al, alo = run_aug_pipeline(x_train, y_train, l_train, aug_proc, aug_mode, aug_pipe_args, seed_number = seed_number)
    logger.info(f"{ad.shape}, {al.shape}")
    
    # Concatenate results
    x_train=np.concatenate((x_train, ad), axis=0)
    y_train=np.concatenate((y_train, al), axis=0)
    l_train = list(l_train) + alo

    # Drop samples having errors.
    dcs = dice_numpy(y_train, y_train, False, 0)
    ixs = np.where(dcs == 1)[0]
    ixs2 = np.where(dcs != 1)[0]

    if len(ixs2) > 0:
        logger.info(f"{len(ixs2)} samples were dropped because their labels have errors.")
        logger.debug(f"The followings were dropped: \nsample_indices:{ixs2}\n{itemgetter(*ixs2)(l_train)}")
        
        plot_prediction_result(x_train[ixs2], y_train[ixs2], y_train[ixs2], list(itemgetter(*ixs2)(l_train)), arrangement = ['d1', 'd2', 'd3', 'd4', 'd1-l1'], save_path = f"{save_dir}/Augmented_samples_having_errors.png")
        x_train, y_train, l_train = x_train[ixs], y_train[ixs], list(itemgetter(*ixs)(l_train))


    logger.info(f"x_train.shape: {x_train.shape}\ny_train.shape: {y_train.shape}")

    del_vars('ad', 'al', 'nii_dict_list', 'nii_dict_list_original', main_global = locals())
    
    return x_train, y_train, l_train


def check_shapes(data, masks = None, logged = None):
    count=0
    r=[]
    u=[]
    shapes=[]

    for ix, i in enumerate(data):
        td = {}

        for j in i:
            tn=(i[j])
            td[j]=tn.shape
            if j == 't1ce': u.append(tn.shape)
            print(f'shape of {j}: ', tn.shape)
        if masks is not None: td['mask'] = masks[ix].shape
        
        if len(set(td.values())) != 1:
            ir={}
            ir['index'] = ix
            if logged is not None: ir['t1ce_path'] = logged[ix]['t1ce'] 
            ir['shapes'] = {k:v for k,v in td.items()}
            r.append(ir)
        
        
        shapes.append(td)
        count +=1
        print(f'Checked count: {count}/{len(data)}')
    #     if count == 5: break

    u=list(set(u))
    
    return r, u, shapes


### Randomly pick data index of selected patients.
def get_index_by_patient_number_old(plist, df1, seed_n = 42, mode = 'random', count_per_patient = 1):
    assert mode in ['random', 'all'], "mode argument should be one of these values: 'random', 'all'."
    
    np.random.seed(seed_n)
    logger.info(f"Numpy seed number was set to {seed_n}.")
    r = []
    for i in plist:
        if mode == 'random':
            r.append(np.random.choice(df1[df1['ID'] == i].index, size = count_per_patient)[0])
        elif mode == 'all':
            r.append(df1[df1['ID'] == i].index)
    
    if mode == 'all': r = list(itertools.chain(*r))
     
    return r

# Plot prediction result
def plot_prediction_result(data, labels, preds, logged = None, chunk_size = 80, save_path = None, arrangement = ['d1', 'd1-l1', 'd1-p1', 'd1-ph', 'd1-l1-ph', 'd1-p1-ph'], **kwargs):
    """
    Plots a comparison figure with arrangement.
    
    Parameters
    ----------
    N : data numbers
    C : channels
    H : Height
    W : Width
    D : Depth
    
    data : a numpy array
        (N, C, H, W, D) array, image data
    
    labels : a numpy array
        (N, C, H, W, D) array, label data
    
    preds : a numpy array
        (N, C, H, W, D) array, preds data
        
    chunk_size : the number of data for one png file.
    
    save_path : png save path.
    
    arrangement : a list of strings
        e.g.
            'd1' means data channel 1
            'd1-l2' means overlaying label channel 2 on data channel 1 
            'd1-p1-ph' means the height is what has the widest prediction area, pred channel 1 overlayed on data channel 1
    
        If you set arrangement to ['d1', 'd1-l1', 'd1-p1', 'd1-ph', 'd1-l1-ph', 'd1-p1-ph'], the figure will be (samples, 6) plot.
            
        
    """
    ca = arange(0, len(data), chunk_size)
    
    for i in range(len(ca)):
        if i == len(ca)-1: break
        i1, i2 = ca[i], ca[i+1]
        
        if logged is None:
            show_mri(data[i1:i2], labels[i1:i2], preds[i1:i2], arrangement = arrangement, initial_num = i1+1, save_path = get_savepath(save_path), **kwargs)
        elif logged is not None:
            show_mri(data[i1:i2], labels[i1:i2], preds[i1:i2], logged=logged[i1:i2], arrangement = arrangement, initial_num = i1+1, save_path = get_savepath(save_path), **kwargs)
        
        
### Plot learning curve for mri data
def plot_learning_curve(history, save_path, 
                        logs_to_plot=['loss', 'val_loss', 'Dec_GT_Output_dice_coefficient', 'val_Dec_GT_Output_dice_coefficient'],
                        initial_epoch=0):
    """
    === For SNUBH MRI Data ===
    plot learning curve and returns val_dice_max_epoch.
    
    
    Parameters
    ----------
    history : log.csv file was written by Csv_Logger of Keras.
    """
    
    history = deepcopy(history)
    history['epoch'] = history['epoch'] + 1
    
    fig, axs = plt.subplots(1,2, figsize=(16,5))

    si=0
    ci=2
    key_3 = re.search(f"^([^\n_]+)", logs_to_plot[2]).group(1)
    
    for ax in axs:
    #     ax.set_xlim(1, len(history[logs_to_plot[0]]))
        for i in logs_to_plot[si:ci]:
            y_max=np.max(history[i])
            x_max=history['epoch'][np.argmax(history[i])]
            y_min=np.min(history[i])
            x_min=history['epoch'][np.argmin(history[i])]

            if re.search('val_loss', i) != None: print(f"val loss min value coordinate: ({x_min}, {y_min})")
            if re.search(f'val.+{key_3}', i) != None:
                print(f"val dice max value coordinate: ({x_max}, {y_max})")
                val_dice_max_epoch = x_max

            ax.plot(history['epoch'], history[i])
            ax.plot(x_max, y_max, 'co', label='_nolegend_') ; ax.plot(x_min, y_min, 'co', label='_nolegend_')
            ax.annotate(f'({x_max}, {y_max:.4f})', xy=(x_max, y_max), xytext=(x_max, y_max + 0.005))
            ax.annotate(f'({x_min}, {y_min:.4f})', xy=(x_min, y_min), xytext=(x_min, y_min - 0.005))

        ax.legend(logs_to_plot[si:ci], bbox_to_anchor=(1,0.5), loc='lower left')
        si += 2 ; ci +=2

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    
    return val_dice_max_epoch


###
def filter_glob_result(glob_result, string = "test"):
    return list(filter(lambda x:re.search(string, x) is None, glob_result))


###
# Save README file
def write_readme(filepath, mode="a"):
    """
    Returns the contents of Readme file.
    
    If some of the variables used in contents string do not exist, this function will raise errors.
    """
    contents=f"""
The Start Date of training : {dp}
Elapsed Time : {tc.show_elapsed_time() if 'tc' in globals() else 'Not exist.'}
TransferLearning or From Scratch : {start_mode if 'start_mode' in globals() else 'Not exist.'}
Shapes of data (N, c, D, H, W) :
    Train data : {x_train.shape}
    Validation data : {x_val.shape}
    Test data : {x_test.shape if 'x_test' in globals() else 'Not exist.'}
Data Augmentation:
    Methods : {aug_proc if 'aug_proc' in globals() else 'Not exist.'}
    Mode : {aug_mode if 'aug_mode' in globals() else 'Not exist.'}
    Pipeline arguments : {aug_pipe_args if 'aug_pipe_args' in globals() else 'Not exist.'}
    Random Invert : {random_invert if 'random_invert' in globals() else 'Not exist.'}
Hyperparameters :
    Initial epoch = {initial_epoch if 'initial_epoch' in globals() else 'Not exist.'}
    Epochs = {epochs if 'epochs' in globals() else 'Not exist.'}
    Learning Rate a0 = {a0 if 'a0' in globals() else 'Not exist.'}
    L2_reg_weight = {l2_reg_weight if 'l2_reg_weight' in globals() else 'Not exist.'}
    Max_norm_value = {max_norm_value if 'max_norm_value' in globals() else 'Not exist.'}
    Dropout_rate = {dropout_rate if 'dropout_rate' in globals() else 'Not exist.'}

Options :
    Monitor_value : {monitor if 'monitor' in globals() else 'Not exist.'}
    EarlyStopping :
        min_delta = {min_delta if 'min_delta' in globals() else 'Not exist.'}
        patience = {patience if 'patience' in globals() else 'Not exist.'}
        baseline = {baseline if 'baseline' in globals() else 'Not exist.'}

Selected Epoch : {val_dice_max_epoch}

Evaluation Results:
    Train GT Dice Score : {train_eval_results[3]:.4f}
    Train VAE Dice Score : {train_eval_results[4]:.4f}

    Validation GT Dice Score : {val_eval_results[3]:.4f}
    Validation VAE Dice Score : {val_eval_results[4]:.4f}

    Test GT Dice Score : {test_eval_results[3]:.4f}
    Test VAE Dice Score : {test_eval_results[4]:.4f}

Comment : {comment}
    """

    with open(filepath, mode) as f:
        f.write(contents)
        
    return contents

### Work in progress ###
def set_insert_position(data, orig_shape):
    """
    returns insert_position array.
    
    Data argument must be rescaled(max value = 255) before being taken by this function.
    """
    
    assert len(data.shape) == 3
    assert len(orig_shape) == 3
    
    r = []
    for i in range(len(orig_shape)):
        while True:
            pos = np.random.uniform(0, 255)
            if pos + data.shape[i] + 1 < orig_shape[i]:
                r.append((slice(pos, pos + data.shape[i] + 1)))
                break
    
    return r


###
def _set_parameters_for_show_mri(img, label, pred, logged, arrangement = ['d1', 'd2', 'd1-l', 'd1-ph', 'd1-ph-p'], Nr=None, Nc=10, figsize=None,
             height=None, cmap='Greys_r', data_order=['x_test', 'y_test', 'y_pred', 'x_test', 'y_pred'], nums = None, dices= None,
             thresh=0.5, save_path = None, pad = 5, subplot_adjust_left = 0.33, max_samples = None, initial_num = None, **kwargs):
    
    # Need to add a section dealing with a situation where pred is not given but arrangement has 'p' string.
    
    nd = len(img)
    Nc = len(arrangement)
    
    if Nr is None: Nr = min(nd, max_samples)
    if figsize is None: figsize = [Nc * 2, Nr * 2]
    if cmap != 'Greys_r': cmap = colors.ListedColormap(cmap)    
    if nums is None: nums = [i+(initial_num-1) for i in range(1, Nr+1)]
    if pred is None: pred = label
    
    # Set dices variable
    try:
        dices = list(map(lambda x,y:dice_numpy(x[None, ...], y[None, ...]), label, pred)) if dices is None else [ -1 ] * min(nd, max_samples)
    except TypeError as e:
        dices = [ -1 ] * min(nd, max_samples)
    
    if logged is not None:
        sample_info = list(map(lambda x:re.sub("^.*([0-9]{8})/([0-9]{4}[-]?[0-9]{2}[-]?[0-9]{2}).*$", "\g<1>\n\g<2>", x), [j['t1'] for j in logged]))
    else:
        sample_info = None
        
    
    lhl = [[np.argmax(np.sum(l[i], axis=(-2,-1))) if height is None else height for i in range(len(l))] for l in label] # label height list 
    phl = [[np.argmax(np.sum(p[i], axis=(-2,-1))) if height is None else height for i in range(len(p))] for p in pred] # prediction height list
    hlr = {'lhl': lhl, 'phl':phl}
    
    color_dict = {'l':[0,1,0], 'p':[0.53, 0.81, 0.92], 'p2':[0.53, 0.81, 0.92]}
    
    # Let's suppose that data_order is ['d1', 'd2', 'd1-l', 'd1-ph', 'd1-ph-p']
    parsed_list = list(map(lambda x:x.split('-'), arrangement))
    chl = list(map(lambda x:int(re.search('d([0-9]+)', x).group(1))-1, arrangement))
    lchl =[] # label channel list
    for e in arrangement:
        sr = re.search('(?:l|p)([0-9]+)', e) # Search result
        if sr is not None:
            lchl.append(int(sr.group(1))-1)
        elif sr is None:
            lchl.append(None)
            
    for i in range(len(lchl)):
        if 'sp' not in locals(): sp = i
        if lchl[i] is None:
            continue
        elif isinstance(lchl[i], int):
            lchl[sp:i] = [lchl[i]] * (i-sp)
            del sp
    
    ll = []
    for e in arrangement:
        if re.search('l([0-9]+)', e) is not None:
            ll.append('l')
        elif re.search('p([0-9]+)', e) is not None:
            ll.append('p')
#         elif 'p2' in e:
#             ll.append('p2')
        else:
            ll.append('nothing')

    hl = []
    for e in parsed_list:
        if 'ph' in e:
            hl.append('phl')
        else:
            hl.append('lhl')
    
    return nd, Nc, Nr, dices, figsize, cmap, lhl, phl, color_dict, chl, ll, hl, hlr, pred, lchl, nums, sample_info


def _plot_the_image(lhl, phl, color_dict, chl, ll, hl, hlr, dices, lchl, nums, sample_info, img_types,
                    img, label, pred, arrangement = ['d1', 'd2', 'd1-l', 'd1-ph', 'd1-ph-p'], Nr=None, Nc=10, figsize=None,
             height=None, cmap='Greys_r', data_order=['x_test', 'y_test', 'y_pred', 'x_test', 'y_pred'],
             thresh=0.5, save_path = None, pad = 5, subplot_adjust_left = 0.33,
                     **kwargs):
    
    # Make subplots
    fig, axs = plt.subplots(Nr, Nc, figsize=figsize)
    if len(axs.shape) == 1: axs = axs.reshape(-1, *axs.shape)
        
    # Plot the image(main part).
    for i in range(Nr):
        hlocs = list(map(lambda x:hlr[x][i], hl)) # height list of current sample
        
        for j in range(Nc):
            axs[i,j].set_xticks([]) ; axs[i,j].set_yticks([])
            #Set an image
            ti = rescale(img[i][chl[j]][hlocs[j][lchl[j]]], 1, (-2, -1))
            ti = np.dstack([ti, ti, ti])
            if ll[j] == 'l':
                m = np.where(label[i][lchl[j]][hlocs[j][lchl[j]]] > thresh)
            elif re.search('p', ll[j]) is not None:
                m = np.where(pred[i][lchl[j]][hlocs[j][lchl[j]]] > thresh)
            if ll[j] in ['l', 'p', 'p2']: ti[m] = color_dict[ll[j]]
        
            axs[i,j].imshow(ti, cmap=cmap, interpolation='none')
    
    logger.debug(f"{dices}, {lhl}, {phl}")
    
    # Set cols list
    try:
        cols = [img_types[int(i)].upper() for i in chl]
    except Exception as e:
        cols = data_order
        
    lchl_mode = max(set(lchl), key=lchl.count)
    if sample_info is not None:
        rows = ['Num {}\n{}\n{:.4f}\n{}\n{}'.format(nums[row], sample_info[row], dices[row], lhl[row][lchl_mode], phl[row][lchl_mode]) for row in range(Nr)]
    else:
        rows = ['Num {}\n{:.4f}\n{}\n{}'.format(nums[row], dices[row], lhl[row][lchl_mode], phl[row][lchl_mode]) for row in range(Nr)]
    
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
#     fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, left=subplot_adjust_left)
    plt.axis('off')
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    return fig, axs


def show_mri(img, label, pred = None, logged = None, arrangement = ['d1', 'd1-l1', 'd1-p1', 'd1-ph', 'd1-ph-l1', 'd1-ph-p1'], Nr=None, Nc=10, figsize=None,
             height=None, cmap='Greys_r', data_order=['x_test', 'y_test', 'y_pred', 'x_test', 'y_pred'], nums = None,
             thresh=0.5, save_path = None, pad = 5, subplot_adjust_left = 0.33, max_samples = 1000, img_types = None, initial_num = 1, **kwargs):
    """
    
    Parameters
    ----------
    N : data numbers
    C : channels
    H : Height
    W : Width
    D : Depth
    
    data : a numpy array
        (N, C, H, W, D) array, image data
    
    labels : a numpy array
        (N, C, H, W, D) array, label data
    
    preds : a numpy array
        (N, C, H, W, D) array, preds data
        
    chunk_size : the number of data for one png file.
    
    save_path : png save path.
    
    arrangement : a list of strings
        e.g.
            'd1' means data channel 1
            'd1-l2' means overlaying label channel 2 on data channel 1 
            'd1-p1-ph' means the height is what has the widest prediction area, pred channel 1 overlayed on data channel 1
    
        If you set arrangement to ['d1', 'd1-l1', 'd1-p1', 'd1-ph', 'd1-l1-ph', 'd1-p1-ph'], the figure will be (samples, 6) plot.
            
        
    """
    
    # Set paramters
    nd, Nc, Nr, dices, figsize, cmap, lhl, phl, color_dict, chl, ll, hl, hlr, pred, lchl, nums, sample_info = _set_parameters_for_show_mri(**locals())
    
    # Plot the image.
    fig, axs = _plot_the_image(**locals())
    
    # Save the image.
    if save_path is not None: fig.savefig(save_path, dpi = 300)


def search_file_w_kw_2(target, keyword, path_pattern='(.+)/.*?pre/.*?$'):
    """
    keyword : a list of keywords.
    path_pattern = a regex pattern which has a group of path in which you want to search files.
    """
    
    r=[]
    cl=[]
    k=f"(?:{str.join('|', keyword)}).*\.nii\.gz"
#     print(k)
    
    for c, i in enumerate(target):
        re_r1 = re.search(path_pattern, i).group(1)
#         print(re_r1)
        gr1 = glob.glob(f"{re_r1}/T1CE*/*.nii.gz") + glob.glob(f"{re_r1}/*.nii.gz") # Edited temporarily, 200323
#         print(gr1)
        ir1 = list(filter(lambda x:re.search(k, x), gr1))
#         print(ir1)
        if len(ir1) == 0:
            ir = [f'Nothing was found. path:{re_r1}']
        else:
            if len(ir1) != 1: cl.append([c, ir1])
            ir=ir1
        r.append(ir)
    
#     r=list(itertools.chain(*r))
    
    return r, cl


### For finding path of snubh dataset
def set_path_of_data(dataset = 'snubh_relab'):
    """
    ### The rules of file names for each scan type and label(snubh_relab)
        - Root path : part3/15324955/2011-05-23/T1CE
        - T1CE : 15324955_2011-05-23_IXI_/BrainExtractionBrain.nii.gz 
        - T2 : 15324955_2011-05-23_IXI_T2_flirt_/IXI_T2_flirt_BrainExtractionBrain.nii.gz
        - Label : (roi|seg|label).nii.gz
    
    if dataset == 'snubh_relab', this returns dataset_path, t1ce, t2, seg, cl.
    elif dataset == 'snuh-bias', this returns dataset_path, t1ce, seg, cl.
    """
    if dataset == 'snubh_relab':
        dataset_path='/data/eck/Workspace/snubh/SNUH_relab' ; logger.info("dataset_path: {}".format(dataset_path))
        os.chdir(dataset_path)

        # Get a list of files for all modalities individually
        root_path='.'
        # t1 = glob.glob(f'{root_path}*t1.nii.gz')
        t2 = glob.glob(f'{root_path}/**/IXI_T2_flirt_BrainExtractionBrain.nii.gz', recursive = True)
        # flair = glob.glob(f'{root_path}*flair.nii.gz')
        t1ce = glob.glob(f'{root_path}/**/*_IXI_/BrainExtractionBrain.nii.gz', recursive=True)
        # seg = glob.glob(f'{root_path}../*seg*', recursive=True)  # Ground Truth

        t2, t1ce = filter_glob_result(t2), filter_glob_result(t1ce)
        
        # Inspect lists of img data
        # Check whether the number of samples are the same over img types.
        len_list = list(map(lambda x:len(globals()[x]), img_types[:-1]))
        print("The counts of each img type: ", len_list)
        if len_list.count(len_list[0]) != len(len_list):
            print("Some scan type images have different numbers of samples.", file=sys.stderr)
        
        seg, cl = search_file_w_kw(t1ce, keyword=['roi', 'seg', 'label'], path_pattern='(.+/[0-9]{4}-[0-9]{2}-[0-9]{2})/.*?$')
        if len(cl) == 0:
            seg = list(chain(*seg))
        
        return dataset_path, t1ce, t2, seg, cl
    
    elif dataset == 'snuh-bias':
        dataset_path='/data/eck/Workspace/snubh/SNUH-bias' ; logger.info("dataset_path: {}".format(dataset_path))
        os.chdir(dataset_path)
            
        # Get a list of files for all modalities individually
        root_path='part*/**/*pre/'
        # t1 = glob.glob(f'{root_path}*t1.nii.gz')
        # t2 = glob.glob(f'{root_path}*t2.nii.gz')
        # flair = glob.glob(f'{root_path}*flair.nii.gz')
        t1ce = glob.glob(f'{root_path}*restore*', recursive=True)
        # seg = glob.glob(f'{root_path}../*seg*', recursive=True)  # Ground Truth
        seg, cl = search_file_w_kw(t1ce, keyword=['roi', 'seg', 'label'], path_pattern='(.+/.*?pre)/.*?$')

        if len(cl) == 0:
            seg = list(chain(*seg))
        
        return dataset_path, t1ce, seg, cl
    
    else:
        raise ValueError("Invaild dataset argument.")
        

def select_time_series(df1, seed_number, count_per_patient):
    np.random.seed(seed_number)
    df1 = df1.copy()
    df1['select_when_random'] = [False] * len(df1)
    
    for pid in df1['ID'].unique():
        ixs = np.random.choice(df1[df1['ID']==pid].index, count_per_patient, replace=False)
        df1.loc[ixs, 'select_when_random'] = True
    
    return df1


### Randomly pick data index of selected patients.
def get_index_by_patient_number(plist, df1, seed_n = 42, mode = 'random', count_per_patient = 1):
    assert mode in ['random', 'all'], "mode argument should be one of these values: 'random', 'all'."
    
    np.random.seed(seed_n)
    logger.info(f"Numpy seed number was set to {seed_n}.")
    r = []
    for i in plist:
        if mode == 'random':
            idt = df1[df1['ID'] == i]
            r.append(np.random.choice(idt[idt['select_when_random'] == True].index, size = count_per_patient).tolist())
        elif mode == 'all':
            r.append(df1[df1['ID'] == i].index)
    
    return list(itertools.chain(*r))


def generate_cross_validation_set(cv_patient_number, df1, split_random_state, train_selection_mode = 'all'):
    for s in cv_patient_number:
        tri = get_index_by_patient_number(s[0], df1 = df1, seed_n = split_random_state, mode = train_selection_mode)
        vai = get_index_by_patient_number(s[1], df1 = df1, seed_n = split_random_state, mode = 'random')
        yield tri, vai
        

def split_data(data, label, logged, split_index, fold_number = None, cvi = None):
    """
    Parameters
    ----------
    fold_number : fold number you want
    cvi : cvi variable which was made at the above.
    split_index : should be [tri, vai, tei].

    """
    tri, vai, tei = split_index
    
    if fold_number is not None: 
        if cvi is None: raise ValueError("cvi argument is also needed when fold_number is given.")
        logger.info(f"fold_number : {fold_number}.")
        tri, vai = cvi[fold_number-1]
    
    x_train, x_val, x_test = data[tri], data[vai], data[tei]
    y_train, y_val, y_test = label[tri], label[vai], label[tei]
    l_train, l_val, l_test = itemgetter(*tri)(logged), itemgetter(*vai)(logged), itemgetter(*tei)(logged)
    
    logger.info(f"{x_train.shape}, {x_val.shape}, {x_test.shape}")
    
    return x_train, x_val, x_test, y_train, y_val, y_test, l_train, l_val, l_test


def truncate_trainset_size(*args, batch_size):
    if all(len(args[0]) == len(i) for i in args) is not True:
        raise ValueError("truncate_trainset_size: Some of inputs have different sizes!")
    
    args = list(args)
    remainder = args[0].shape[0] % batch_size
    if remainder != 0:
        for i,v in enumerate(args):
            args[i] = v[:len(v)-remainder]
        logger.info(f"trainset size was modified from {args[0].shape[0]+remainder} to {args[0].shape[0]}.")
        
    return args


def make_model(build_model, input_shape, output_channels, n_gpu, test_mode = True, seed_number = 42):
    tf.random.set_seed(seed_number)
    np.random.seed(seed_number)
    logger.info(f"{seed_number} was set to seed number, and seed of tensorflow and numpy was set to the number.")

    model, opt, lg, lv, dc = build_model(input_shape=input_shape,
                                                    output_channels=output_channels, n_gpu=n_gpu, test_mode=True)
    
    return model, opt, lg, lv, dc


def set_callbacks(model, save_dir, fold_number = None, min_delta = 0.01, patience = 10, baseline = None, a0 = 1e-5, lr_schedule_total_epoch = 300,
                  initial_epoch = None,
                  monitor='val_Dec_GT_Output_loss',
                  filename_w=None, csv_prefix = ""):
    """
    Parameters
    ----------
    save_dir
    fold_number
    initial_epoch : when manually select initial epoch.
    
    === Early Stopping ===
    min_delta :
    patience : 
    baseline :
    
    === Learning Rate Schedule ===
    a0 :
    lr_schedule_total_epoch :
    
    === Model Checkpoint ===
    monitor :
    filename_w :
    
    Returns
    -------
    tensorboard_cb, customlogs_cb, checkpoint_cb_w, earlystopping_cb, lr_scheduler, cvlogger, csv_logger
    
    """
    
    fold_dir = set_fold_dir(save_dir, fold_number)
    
    if filename_w is None: filename_w = "keras_weight_epoch-{epoch:02d}_" + f"{monitor}" + "-{" + f"{monitor}" +":.4f}.h5"
    
    # tensorboard
    root_logdir = os.path.join(os.curdir, "my_logs")

    def get_run_logdir(): 
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir() # e.g., './ my_logs/ run_2019_06_07-15_15_22'

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    customlogs_cb = CustomLogs() # Custom Logs to record elapsed time.
    cvlogger = CrossValidationLogger(f'{save_dir}/cv_log.csv', fold_number, monitor, append = True) # CrossValidation Logger
    
    # ModelCheckpoint
    logger.info(f"ModelCheckpoint monitor value: {monitor}")

    checkpoint_path_w=os.path.join(fold_dir, filename_w)

    checkpoint_cb_w = keras.callbacks.ModelCheckpoint(checkpoint_path_w, monitor=monitor, save_weights_only=True)
    
    # early stopping
    logger.info(f"min_delta: {min_delta}")
    logger.info(f"patience: {patience}")
    logger.info(f"baseline: {baseline}")
    earlystopping_cb = keras.callbacks.EarlyStopping(monitor=monitor, min_delta = min_delta,
                                                     patience = patience, baseline = baseline)
    
    # Set initial Epoch
    log_csv_path = f"{fold_dir}/{csv_prefix}log.csv"
    
    try:
        log_csv = pd.read_csv(log_csv_path)
    except Exception as e:
        logger.info(f"Failed to load {log_csv_path}.\n{e.__class__.__name__}: {e}")
        
    if initial_epoch is None:
        try:
            last_epoch = log_csv['epoch'].max() + 1 ; logger.info(f"last epoch = {last_epoch}")
            ftl = glob.glob(f"{fold_dir}/*weight*epoch-{last_epoch:02d}*.h5")[0]
            model.load_weights(ftl) ; logger.info(f"{ftl} file was loaded to the model.")
            initial_epoch = last_epoch
        except Exception as e:
            initial_epoch = 0 ; logger.info(f"{e.__class__.__name__}: {e}\nInitial_epoch was set to {initial_epoch}.")
    
    elif isinstance(initial_epoch, int):
        # Set Initial Epoch Manually.
        try:
            log_csv.to_csv(get_savepath(re.sub(f"log\.csv", f"log_backup.csv", log_csv_path)), index=False)
            if initial_epoch != 0:
                log_csv.iloc[:initial_epoch, :].to_csv(log_csv_path, index=False)
            elif initial_epoch == 0:
                os.remove(log_csv_path)
            
            # load saved weights if initial_epoch is not 0.
            ftl = glob.glob(f"{fold_dir}/*weight*epoch-{initial_epoch:02d}*.h5")[0]
            model.load_weights(ftl); logger.info(f"{ftl} file was loaded to the model.")
            
        except Exception as e:
            logger.info(f"{e.__class__.__name__}: {e}\nInitial_epoch was set to {initial_epoch}.")
            # Csv logger
            csv_logger = keras.callbacks.CSVLogger(log_csv_path, append = True)
    
    # Learning Rate Schedule
    logger.info("a0 was set to {}".format(a0))
    logger.info("lr_schedule_total_epoch for lr_schedule was set to {}".format(lr_schedule_total_epoch))
    def lr_schedule(a0=1e-4, total_epoch=300):

        def lr_schedule_(epoch):
            a=a0*(1-epoch/total_epoch)**0.9

            return a

        return lr_schedule_

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule(a0, lr_schedule_total_epoch))
    
    # Csv logger
    csv_logger = keras.callbacks.CSVLogger(log_csv_path, append = True)
    
    # Log logger
    log_logger = LogLogger(logger)
    
    return initial_epoch, tensorboard_cb, customlogs_cb, checkpoint_cb_w, earlystopping_cb, lr_scheduler, cvlogger, log_logger, csv_logger


def fit_model(model, x_train, y_train, batch_size, epochs, initial_epoch, callbacks = [], x_val=None, y_val=None, steps_per_epoch = None):
    tc = TimeChecker()
    
    # Set validation set
    val_set = (x_val, [y_val, x_val]) if x_val is not None else None
    
    logger.info(f"Start training... epochs = {epochs}, batch_size = {batch_size}.")
    history = model.fit(x_train, [y_train, x_train], validation_data = val_set, 
                        batch_size=batch_size, epochs = epochs, initial_epoch=initial_epoch, steps_per_epoch = steps_per_epoch,
                        callbacks = callbacks) # Default callback count : 6
    
    tc.set_end()
    
    return history, tc


def set_fold_dir(save_dir, fold_number):
    if isinstance(fold_number, int):
        fold_dir = f"{save_dir}/CV_results/{fold_number}-fold"
        os.makedirs(fold_dir, exist_ok = True)
    else:
        fold_dir = save_dir
        
    logger.debug(f"fold_dir was set to {fold_dir}.")
    
    return fold_dir


def make_ber_df(best_epoch, cvi, save_dir):
    """
    Make a dataframe of best epoch result.
    """
    for k in range(1, len(cvi)+1):
        temp_csv = pd.read_csv(f"{save_dir}/CV_results/{k}-fold/log.csv")
        temp_csv = temp_csv[temp_csv['epoch'] == best_epoch-1]

        if k == 1:
            ber_df = pd.DataFrame(index = pd.Index([f"{k}-fold" for k in range(1, len(cvi)+1)], name = "fold_number"),
                                 columns = pd.Index(temp_csv.columns.values, name = "Values"))

        ber_df.loc[f"{k}-fold",] = temp_csv.values

    ber_df['epoch'] += 1
    ber_df.loc['mean'] = ber_df.mean(axis=0)
    ber_df.to_csv(f"{save_dir}/ber_df.csv")
    
    return ber_df


def cross_val_process(data, label, logged, split_index, 
                           cvi, aug_proc,
                           aug_mode, aug_pipe_args, build_model, input_shape, output_channels, n_gpu, save_dir,
                           epochs, batch_size, data_dtype = np.float32, start_fold_number = 1,
                           seed_number = 42, test_mode = False, a0 = 1e-5, lr_schedule_total_epoch= 50, source_model_weight = "/data/eck/Workspace/snubh/200325_1820_Brats_T1_source_model/keras_weight_epoch-32_val_Dec_GT_Output_loss--0.7759.h5",
                           monitor = "val_Dec_GT_Output_loss"
                     ):
    """
    Parameters
    ----------
    split_index : should be [tri, vai, tei].

    """
    
    tc_main = TimeChecker()
    
    for fold_number in range(start_fold_number, len(cvi)+1):
        fold_dir = set_fold_dir(save_dir, fold_number)
        
        x_train, x_val, x_test, y_train, y_val, y_test, l_train, l_val, l_test = split_data(data, label, logged, split_index, fold_number, cvi)
        logger.info(f"Data was split.")
        
        if test_mode is True: x_train, y_train = x_train[:1], y_train[:1]
        
        if len(aug_proc) > 0:
            x_train, y_train, l_train = data_augmentation(x_train, y_train, l_train, aug_proc, aug_mode, aug_pipe_args, save_dir, seed_number = seed_number, data_dtype = np.float32)
            logger.info(f"Data was augmented.")
        
        x_train, y_train, l_train = truncate_trainset_size(x_train, y_train, l_train, batch_size = batch_size)
        
        logger.info(f"\nTrain set size:{x_train.shape[0]}\nValidation set size: {x_val.shape[0]}")
        model, template_model, opt, lg, lv, dc = make_model(build_model, input_shape, output_channels, n_gpu, test_mode = True, seed_number = seed_number)

        template_model.layers[-2].name = 'Dec_GT_Output_1c'
        template_model.load_weights(source_model_weight, by_name = True)

        logger.info(f"Model was created.")
        
        initial_epoch, tensorboard_cb, customlogs_cb, checkpoint_cb_w, earlystopping_cb, lr_scheduler, cvlogger, log_logger, csv_logger = set_callbacks(model, save_dir, fold_number = fold_number, min_delta = 0.01, patience = 10, baseline = None, a0 = a0, lr_schedule_total_epoch = lr_schedule_total_epoch, initial_epoch=None, monitor=monitor)
        
        callbacks = [tensorboard_cb, customlogs_cb, checkpoint_cb_w, lr_scheduler, cvlogger, log_logger, csv_logger]
        
        history, tc = fit_model(model, x_train, y_train, batch_size, epochs, initial_epoch, callbacks, x_val, y_val)        
        
        pickle.dump([x_train, y_train, l_train], open(f"{fold_dir}/train_set.pkl", "wb"), protocol=4)
        pickle.dump([x_val, y_val, l_val], open(f"{fold_dir}/val_set.pkl", "wb"), protocol=4)
        plot_learning_curve(pd.read_csv(f"{fold_dir}/log.csv"), save_path=get_savepath(f"{save_dir}/{fold_number}-fold_cv_learning_curve.png"))
        
        logger.info(f">> {fold_number}-fold complete.")
        
    logger.info(f"Elapsed time: {tc_main.set_and_show()}.")
    cvlogger.df['mean'] = cvlogger.df.mean(axis=1)
    cvlogger.df.to_csv(f'{save_dir}/cv_log.csv')
    tdf1 = cvlogger.df
    if tdf1.mean(axis=1).sum() > 0: tdf1 *= -1
    best_epoch = tdf1.mean(axis=1).idxmin() ; logger.info(f"The best epoch : {best_epoch}, mean score: {cvlogger.df.mean(axis=1).min()}")
    best_cv_score = tdf1.mean(axis=1)[best_epoch]
    make_ber_df(best_epoch, cvi, save_dir)
    
    # Record prediction result
    cross_val_predict(model, best_epoch, batch_size, cvi, save_dir, data_kind = 'train')
    cross_val_predict(model, best_epoch, batch_size, cvi, save_dir, data_kind = 'val')
    
    return best_epoch, best_cv_score, model


def cross_val_predict(model, best_epoch, batch_size, cvi, save_dir, data_kind):
    imgs = []
    labels = []
    preds = []
    l_vals = []
    
    if data_kind not in ['train', 'val']: raise ValueError(f"data_kind argument should be one of the followings: 'train', 'val'")
    
    for fold_number in range(1, len(cvi)+1):
        fold_dir = set_fold_dir(save_dir, fold_number)
        
        data_pkl_path = f"{fold_dir}/{data_kind}_set.pkl"
        
        x_val, y_val, l_val = pickle.load(open(data_pkl_path, "rb"))
        load_weights_of_epoch(model, best_epoch, fold_dir)
        
        y_val_pred, _ = model.predict(x_val, batch_size = batch_size * 3)
        
        plot_prediction_result(x_val, y_val, y_val_pred, l_val, chunk_size = 80, save_path = f"{save_dir}/{fold_number}-fold_{data_kind}-preds_plot.png")
        
        pickle.dump([x_val, y_val, y_val_pred, l_val], open(f"{save_dir}/{fold_number}-fold_{data_kind}_preds.pkl", "wb"), protocol=4)
        logger.info(f"[images, labels, preds, logged] was saved to {save_dir}/{fold_number}-fold_{data_kind}_preds.pkl")
        
        os.remove(data_pkl_path)
        
        if data_kind == 'val':
            imgs.append(x_val)
            labels.append(y_val)
            preds.append(y_val_pred)
            l_vals.append(l_val)
    
    if data_kind == 'val':
        imgs = np.concatenate(imgs, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        l_vals = list(chain(*l_vals))
        
        plot_prediction_result(imgs, labels, preds, l_vals, chunk_size = 80, save_path = f"{save_dir}/cross-val_{data_kind}-preds_plot.png")
        pickle.dump([imgs, labels, preds, l_vals], open(f"{save_dir}/cross_val_preds.pkl", "wb"), protocol = 4)

    logger.info(f"[images, labels, preds, logged] were saved to {save_dir}/cross_val_preds.pkl.")

    return imgs, labels, preds, l_vals
    

###
class spm12_input_dict:
    def __init__(self, root_folder, img_types, pid = None, file_extension = ".nii"):
        self.root_folder = os.path.abspath(root_folder)
        self.img_types = img_types
        self.file_extension = file_extension
        self.fep = re.sub("\.", "\\.", file_extension) # file_extension_pattern
        self.set_input_dict(root_folder, img_types, pid)
        
        
    def __str__(self):
        return self.values.__str__()
    
    def __repr__(self):
        return self.values.__repr__()
    
    def set_input_dict(self, root_folder, img_types, pid):
        root_folder = os.path.abspath(root_folder)
        tfl1 = glob.glob(f"{root_folder}/**/*{self.file_extension}", recursive = True) # temp folder list 1
        
        # modify pid variable.
        if (pid == 'all') or (pid == ['all']):
            pid = set(if_found_return_groups("[0-9]{8}", tfl1, 0))
        elif isinstance(pid, int):
            pid = [str(pid)]
        elif pid is None:
            raise ValueError(f"pid should be given.")

        self.pid = pid
        input_dict = dict.fromkeys(self.pid)
        
        for i in input_dict:
            gr1 = glob.glob(f"{root_folder}/**/{i}/**/*{self.file_extension}", recursive = True)
            if len(gr1) == 0: raise ValueError(f"gr1 has nothing. {i}\n{gr1}")
            
            if self.file_extension != "_final.nii":
                gr1 = list(filter(lambda x:re.search("/(?:[rmc]{1,2}|mean).*" + self.fep, x) is None, gr1))
                gr1 = list(filter(lambda x:re.search("/.*_final" + self.fep, x) is None, gr1))
            else:
                pass
            
            if len(gr1) == 0: raise ValueError(f"gr1_2 has nothing. {i}\n{gr1}")
            
            ifr1 = set(if_found_return_groups("([0-9]{4}-[0-9]{2}-[0-9]{2})", gr1, group_index = 1))
            if len(ifr1) == 0: raise ValueError(f"ifr1 has nothing. {i}\nifr1:{ifr1}\ngr1:{gr1}")
            
            input_dict[i] = dict.fromkeys(ifr1)
            
            for j in input_dict[i]:
                td = {}
                    
                for t in img_types:
                    if t != 'seg':
                        tt= 't1' if t=='t1ce' else t
                        tl = if_found_return_groups("^((?!roi|seg|label).)*$",
                                                        if_found_return_groups(f"{root_folder}(?:/[^/\n]+)*/{i}/{j}.*/[^/\n]*{tt}[^/\n]*{self.fep}", gr1, 0, re.I),
                                                        0,
                                                        re.I)
                        if len(tl) == 1:
                            td[t] = tl[0]
                        else:
                            raise ValueError(f"More than one path was found. {tl} {i} {j} {t} {tt} {gr1}")

                    else:
                        tl = if_found_return_groups(f"{root_folder}(?:/[^/\n]+)*/{i}/{j}.*/[^/\n]*(?:roi|seg|label)[^/\n]*{self.fep}", gr1, 0, re.I)
                        if len(tl) == 1:
                            td[t] = tl[0]
                        else:
                            raise ValueError(f"More than one path was found. {tl} {i} {j} {t} {tt} {gr1}")

                input_dict[i][j] = td
            
        self.values = input_dict
        
        
def move_needless_weight_files(save_dir, selected_epoch, dest_dir="trashcan"):
    wh5l = glob.glob(f"{save_dir}/*weight*.h5")
    last_epoch = max(list(map(lambda x:int(re.search("epoch-([0-9]+)", x).group(1)), wh5l)))
    files_to_delete = list(filter(lambda x:re.search(f"epoch-(?:{last_epoch:02d}|{selected_epoch:02d})", x) is None, wh5l))
    for f in files_to_delete:
        copy_keeping_structure(f, save_dir, f"{dest_dir}/{save_dir}", copy_function = shutil.move) ; logger.info(f"{f} was moved to trashcan folder.")
'''
def downsize_and_padding(data, label, dp_factors_range = (0.2, 0.8)):
    assert len(data.shape) == 4, "The ndim of data should be 4."
    
    data = rescale(data, 255)
    factors = [np.random.uniform(*dp_factors_range)] * 3
    
    # process data
    ir = []
    for d in data:
        za = np.zeros(d.shape)
        d = zoom(d, factors, order = 0)
        try:
            insert_pos = set_insert_position
        za[insert_pos] = d
        ir.append(za)
    
    # process label
    za = np.zeros(label.shape)
    l = zoom(label, factors, order = 0)
    za[insert_pos] = l
    label = za
    
    return [[np.stack(ir), label]]
'''


def arrangement_data(order_list, logged):
    ol=order_list
    
    ixs=[]
    for i in order_list:
        print(i)
        tl1 = list(
            map(lambda x:True if re.search(i, x) is not None else None, [j['t1'] for j in logged])
        )
        
        try:
            ixs.append(tl1.index(True))
        except ValueError as e:
            print(i, e.__class__.__name__, e)
    
    return ixs


###
def get_index_by_sample_info(pid, date, logged):
    """
    Parameters
    ----------
    pid
    date
    logged
    """
    
    date_pattern = re.sub("([0-9]{4})[-]?([0-9]{2})[-]?([0-9]{2})", "\g<1>[-]?\g<2>[-]?\g<3>", str(date))
    fl = []
    for i in logged:
        fl.append(
                re.search(f"{pid}/{date_pattern}", i['t1']) is not None
        )
        
    return [i for i,v in enumerate(fl) if v is True]