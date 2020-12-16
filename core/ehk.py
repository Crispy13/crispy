import cv2
import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from PIL import Image
from matplotlib import cm, colors
import crispy as csp
import tensorflow as tf
from tensorflow import keras
import tifffile
import pandas as pd
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import tensorflow_addons as tfa

from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

Image.MAX_IMAGE_PIXELS = None


###
def load_tiff(path):
    image = tifffile.imread(path)
    
    if image.shape[-1] != 3:
        print("The shape of loaded image will be reshaped, current shape = {}".format(image.shape))
        image = np.squeeze(image)
        image = np.transpose(image, axes = (1, 2, 0))
        
    print("Sample id: {}, image shape = {}".format(Path(path).stem, image.shape))
    
    return image


###
def make_mask(json_path, image):
    json_data = json.load(open(json_path, "r"))
    
    ## Make polygons
    polys = []
    for index in range(json_data.__len__()):
        geom = np.array(json_data[index]['geometry']['coordinates'])
        polys.append(geom)
    
    mask = np.zeros(image.shape[:-1])
    mask = np.expand_dims(mask, axis = -1)
    
    for i in range(len(polys)):
        cv2.fillPoly(
                        mask, polys[i], 
                        1
        )
    
    mask = mask.astype(bool)
    
    print("Mask shape: {}".format(mask.shape))
    
    return mask


###
def get_tile(baseimage, tile_size, tile_row_pos, tile_col_pos, stride):
    start_col = tile_col_pos * stride
    end_col = start_col + tile_size
    start_row = tile_row_pos * stride
    end_row = start_row + tile_size
    tile_image = baseimage[start_row:end_row, start_col:end_col, :]
    
    
    ## For truncated tiles, pad zeros to the tiles in order to get the same shape as normal tiles.
    if tile_image.shape != (tile_size, tile_size, baseimage.shape[-1]):
        zero_array = np.zeros((tile_size, tile_size, baseimage.shape[-1]))
        row, col, ch = tile_image.shape
        zero_array[:row, :col, :ch] = tile_image
        
        tile_image = zero_array
        orig_tile_shape = (row, col, ch)
        
    else:
        orig_tile_shape = "no"
    
    tile_image = tile_image.astype(np.uint8)
    
    return tile_image, orig_tile_shape


###
def show_tile_and_mask(tile_image, tile_mask):
    fig, ax = plt.subplots(1,2,figsize=(20,3))
    ax[0].imshow(tile_image)
    ax[1].imshow(tile_mask)
    
    
# Utilities serialize data into a TFRecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


###
def image_example(image_index, image, mask, tile_id, tile_col_pos, tile_row_pos):
    image_shape = image.shape
    
    img_bytes = image.tobytes()

    mask_bytes = mask.tobytes()
    
    feature = {
        'img_index': _int64_feature(image_index),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'num_channels': _int64_feature(image_shape[2]),
        'img_bytes': _bytes_feature(img_bytes),
        'img_dtype': _bytes_feature(str(image.dtype).encode()),
        'mask' : _bytes_feature(mask_bytes),
        'mask_dtype' :  _bytes_feature(str(mask.dtype).encode()),
        'tile_id':  _int64_feature(tile_id),
        'tile_col_pos': _int64_feature(tile_col_pos),
        'tile_row_pos': _int64_feature(tile_row_pos),
        
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


###
def create_tfrecord(image_index, image, mask, tile_id, tile_col_pos, tile_row_pos, output_path):
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(str(output_path), opts) as writer:
        tf_example = image_example(image_index, image, mask, tile_id, tile_col_pos, tile_row_pos)
        writer.write(tf_example.SerializeToString())
    writer.close()
    
    
    
###
def write_tfrecord_tiles(image_index, image_id, image, mask, tile_size, stride, output_path):
    output_dir = Path(output_path) / image_id
#     if output_dir.exists():
#         shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok = True)
    
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    tile_rows = (image_rows-1) // stride + 1
    tile_cols = (image_cols-1) // stride + 1
    tileID = 0
    
    pb = csp.Progressbar(total = tile_rows * tile_cols)
    
    # create a pandas dataframe to store metadata for each tile
    tile_df = pd.DataFrame(columns = ['img_index', 'img_id', 'tile_id', 'tile_rel_path', 'tile_row_num', 'tile_col_num', 'lowband_density', 'mask_density', "zero_padded"])
    
    # create one directory for each row of images
    for row_number in range(tile_rows):
#         print('row_offset{} '.format(row_number),end='')
#         dir_path = output_dir / 'row{}'.format(row_number)
#         # create directory
#         if dir_path.exists():
#             shutil.rmtree(dir_path)
#         dir_path.mkdir()
        for col_number in range(tile_cols):
            #print("row{}".format(col_number),end='')
#             dataset_file_path = dir_path+'/col{}_row{}.tfrecords'.format(row_number,col_number)
            
            dataset_file_path = dir_path / 'row{}_col{}.tfrecords'.format(row_number, col_number)
            relative_path = image_id + "/row{}_col{}.tfrecords".format(row_number, col_number)
#             if dataset_file_path.exists():
#                 shutil.rmtree(dataset_file_path)
#             dataset_file_path.mkdir(parents = True, exist_ok = True)
        
            lower_col_range = col_number * stride
            image_tile, orig_image_shape = get_tile(image, tile_size, row_number, col_number, stride)
            tile_mask, _ = get_tile(image, tile_size, row_number, col_number, stride)
            num_records = create_tfrecord(image_index, image_tile, tile_mask, tileID, row_number, col_number, dataset_file_path)
            
            # populate the metadata for this tile
            img_hist = np.histogram(image_tile)
            lowband_density = np.sum(img_hist[0][0:4])
            mask_density = np.count_nonzero(tile_mask)
            tile_df = tile_df.append({'img_index':image_index, 'img_id':image_id, 'tile_id': tileID, 'tile_rel_path':relative_path, 
                           'tile_col_num':col_number, 'tile_row_num':row_number,'lowband_density':lowband_density, 'mask_density':mask_density, "zero_padded":orig_image_shape},ignore_index=True)
            
            pb.show(tileID, details = "Current sample id = {}, row_offset = {}, col_offset = {}".format(image_id, row_number, col_number))
            tileID += 1
            
    return tile_df


###
def write_tfrecord_tiles_mp(image_index, image_id, image, mask, tile_size, stride, output_path):
    output_dir = Path(output_path) / image_id
#     if output_dir.exists():
#         shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok = True)
    
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    tile_rows = (image_rows-1) // stride + 1
    tile_cols = (image_cols-1) // stride + 1
    
    write_tfrecord_tiles_mp_main_partial = partial(write_tfrecord_tiles_mp_main, 
                                                    image_index = image_index, image_id = image_id, output_dir = output_dir, 
                                                   tile_size = tile_size, stride = stride, tile_cols = tile_cols
                                                  )
    
    v.map(os.chdir, [os.getcwd()] * len(v))
    r = v.map_async(write_tfrecord_tiles_mp_main_partial, list(range(tile_rows)))
    print(dir(r))
#     csp.track_job(r, tile_rows)
    r.wait()
    ipc_result = r.get()
    
    print("Multiprocessing job done.")


#     with Pool(cpu_count()) as p:
#         r = p.map_async(write_tfrecord_tiles_mp_main_partial, list(range(tile_rows)))
#         csp.track_job(r, tile_rows)
        
#         r.wait()
        
#         pool_result = r.get()
        
#         print("Multiprocessing job done.")


###
def write_tfrecord_tiles_mp_main(row_number, image, mask, image_index, image_id, output_dir, tile_cols, tile_size, stride):
#     print("write_tfrecord_tiles_mp_main: starting...")
    tile_df = pd.DataFrame(columns = ['img_index', 'img_id', 'tile_id', 'tile_rel_path', 'tile_row_num', 'tile_col_num', 'lowband_density', 'mask_density', "zero_padded"])
    tileID = 0
    dir_path = Path(output_dir) / "row{}".format(row_number)
    dir_path.mkdir(exist_ok = True)
    
#     print("write_tfrecord_tiles_mp_main: Entering for loop ...")
    for col_number in range(tile_cols):
        print("Starting to write tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        #print("row{}".format(col_number),end='')
        dataset_file_path = dir_path / 'row{}_col{}.tfrecord'.format(row_number, col_number)
        relative_path = image_id + "/row{0}/row{0}_col{1}.tfrecord".format(row_number, col_number)

        lower_col_range = col_number * stride
        image_tile, orig_image_shape = get_tile(image, tile_size, row_number, col_number, stride)
        tile_mask, _ = get_tile(mask, tile_size, row_number, col_number, stride)
        num_records = create_tfrecord(image_index, image_tile, tile_mask, tileID, row_number, col_number, dataset_file_path)

        # populate the metadata for this tile
        img_hist = np.histogram(image_tile)
        lowband_density = np.sum(img_hist[0][0:4])
        mask_density = np.count_nonzero(tile_mask)
        tile_df.loc[tileID, :] = {'img_index':image_index, 'img_id':image_id, 'tile_id': tileID, 'tile_rel_path':relative_path, 
                       'tile_col_num':col_number, 'tile_row_num':row_number,'lowband_density':lowband_density, 'mask_density':mask_density, "zero_padded":orig_image_shape}

        tileID += 1
        
        del image_tile, tile_mask, orig_image_shape
        
        print("Complete writing tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        
    tile_df.to_csv(dir_path / "{}_row{}_tile-df.csv".format(image_id, row_number))
    print("Complete writing tfrecords... {}".format(image_id))
    
    
# read back a record to make sure it the decoding works
# Create a dictionary describing the features.
image_feature_description = {
    'img_index': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'num_channels': tf.io.FixedLenFeature([], tf.int64),
    'img_bytes': tf.io.FixedLenFeature([], tf.string),
    'img_dtype': tf.io.FixedLenFeature([], tf.string),
    'mask': tf.io.FixedLenFeature([], tf.string),
    'mask_dtype': tf.io.FixedLenFeature([], tf.string),
    'tile_id': tf.io.FixedLenFeature([], tf.int64),
    'tile_col_pos': tf.io.FixedLenFeature([], tf.int64),
    'tile_row_pos': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    img_index = single_example['img_index']
    img_height = single_example['height']
    img_width = single_example['width']
    num_channels = single_example['num_channels']
    
    img_dtype = tf.io.decode_raw(single_example['img_dtype'], out_type = tf.string)
    img_bytes =  tf.io.decode_raw(single_example['img_bytes'], out_type = img_dtype)
   
    img_array = tf.reshape( img_bytes, (img_height, img_width, num_channels))
       
    mask_dtype = tf.io.decode_raw(single_example['mask_dtype'], out_type = tf.string)
    mask_bytes =  tf.io.decode_raw(single_example['mask'], out_type = mask_dtype)
    
    mask = tf.reshape(mask_bytes, (img_height,img_width, 1))

    mtd = dict()
    mtd['img_index'] = single_example['img_index']
    mtd['width'] = single_example['width']
    mtd['height'] = single_example['height']
    mtd['tile_id'] = single_example['tile_id']
    mtd['tile_col_pos'] = single_example['tile_col_pos']
    mtd['tile_row_pos'] = single_example['tile_row_pos']
    struct = {
        'img_array': img_array,
        'mask': mask,
        'mtd': mtd
    } 
    return struct

def read_tf_dataset(storage_file_path):
    encoded_image_dataset = tf.data.TFRecordDataset(storage_file_path, compression_type="GZIP")
    parsed_image_dataset = encoded_image_dataset.map(_parse_image_function)
    return parsed_image_dataset



###
def write_tfrecord_tiles_mp_main_test(row_number, trainset_dir, image_index, image_id, output_dir, tile_cols, tile_size, stride):
    image = load_tiff(trainset_dir / "{}.tiff".format(image_id))
    mask = make_mask(trainset_dir / "{}.json".format(image_id), image)
    
#     print("write_tfrecord_tiles_mp_main: starting...")
    tile_df_path = dir_path / "{}_row{}_tile-df.csv".format(image_id, row_number)
    if tile_df_path.exists():
        tile_df = pd.read_csv(tile_df_path, index_col = 0)
    else:
        tile_df = pd.DataFrame(columns = ['img_index', 'img_id', 'tile_id', 'tile_rel_path', 'tile_row_num', 'tile_col_num', 'lowband_density', 'mask_density', "zero_padded"])
    
    tileID = 0
    dir_path = Path(output_dir) / "row{}".format(row_number)
    dir_path.mkdir(exist_ok = True)
    
#     print("write_tfrecord_tiles_mp_main: Entering for loop ...")
    for col_number in range(tile_cols):
        print("Starting to write tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        #print("row{}".format(col_number),end='')
        dataset_file_path = dir_path / 'row{}_col{}.tfrecord'.format(row_number, col_number)
        relative_path = image_id + "/row{0}/row{0}_col{1}.tfrecord".format(row_number, col_number)

        lower_col_range = col_number * stride
        image_tile, orig_image_shape = get_tile(image, tile_size, row_number, col_number, stride)
        tile_mask, _ = get_tile(mask, tile_size, row_number, col_number, stride)
        num_records = create_tfrecord(image_index, image_tile, tile_mask, tileID, row_number, col_number, dataset_file_path)

        # populate the metadata for this tile
        img_hist = np.histogram(image_tile)
        lowband_density = np.sum(img_hist[0][0:4])
        mask_density = np.count_nonzero(tile_mask)
        tile_df.loc[tileID, :] = {'img_index':image_index, 'img_id':image_id, 'tile_id': tileID, 'tile_rel_path':relative_path, 
                       'tile_col_num':col_number, 'tile_row_num':row_number,'lowband_density':lowband_density, 'mask_density':mask_density, "zero_padded":orig_image_shape}

        tileID += 1
        
        del image_tile, tile_mask, orig_image_shape
        
        print("Complete writing tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        
    
    tile_df_path.exist
    tile_df.to_csv()
    print("Complete writing tfrecords... {}".format(image_id))
    
    
    
###
def write_tfrecord_tiles_mp_main_test2(row_number, image_index, image_id, output_dir, tile_cols, tile_size, stride):
#     print("write_tfrecord_tiles_mp_main: starting...")

    dir_path = Path(output_dir) / "row{}".format(row_number)
    dir_path.mkdir(exist_ok = True)
    print(id(image), id(mask))
    
    ## set tile_df
    tile_df_path = dir_path / "{}_row{}_tile-df.csv".format(image_id, row_number)
    
    if tile_df_path.is_file():
        tile_df = pd.read_csv(tile_df_path, index_col = 0)
    else:
        tile_df = pd.DataFrame(columns = ['img_index', 'img_id', 'tile_id', 'tile_rel_path', 'tile_row_num', 'tile_col_num', 'lowband_density', 'mask_density', "zero_padded"])
        
    tileID = 0
    
#     print("write_tfrecord_tiles_mp_main: Entering for loop ...")
    for col_number in range(tile_cols):
#         print("Starting to write tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        #print("row{}".format(col_number),end='')
        dataset_file_path = dir_path / 'row{}_col{}.tfrecord'.format(row_number, col_number)
        relative_path = image_id + "/row{0}/row{0}_col{1}.tfrecord".format(row_number, col_number)

        lower_col_range = col_number * stride
        image_tile, orig_image_shape = get_tile(image, tile_size, row_number, col_number, stride)
        tile_mask, _ = get_tile(mask, tile_size, row_number, col_number, stride)
        
        print("image_tile.shape = {}\nimage_id = {}, row_number = {}, col_number = {}".format(image_tile.shape, image_id, row_number, col_number))
        if image_tile.shape != (tile_size, tile_size, 3):
            raise ValueError("tile size is not {} but {}\nimage_id = {}, row_number = {}, col_number = {}".format((tile_size, tile_size, 3), image_tile.shape, image_id, row_number, col_number))
        
        num_records = create_tfrecord(image_index, image_tile, tile_mask, tileID, row_number, col_number, dataset_file_path)

        # populate the metadata for this tile
        img_hist = np.histogram(image_tile)
        lowband_density = np.sum(img_hist[0][0:4])
        mask_density = np.count_nonzero(tile_mask)
        tile_df.loc[tileID, :] = {'img_index':image_index, 'img_id':image_id, 'tile_id': tileID, 'tile_rel_path':relative_path, 
                       'tile_col_num':col_number, 'tile_row_num':row_number,'lowband_density':lowband_density, 'mask_density':mask_density, "zero_padded":orig_image_shape}

        tileID += 1
        
        del image_tile, tile_mask, orig_image_shape
        
#         print("Complete writing tfrecords... {}-row{}-col{}".format(image_id, row_number, col_number))
        
    tile_df.to_csv(tile_df_path)
#     print("Complete writing tfrecords... {}".format(image_id))
    
    
###    
def pool_init2(image_base, mask_base, image_shape, mask_shape):
    global image, mask
    image = np.ctypeslib.as_array(image_base.get_obj())
    image = image.reshape(*image_shape)
   
    mask = np.ctypeslib.as_array(mask_base.get_obj())
    mask = mask.reshape(*mask_shape)
    
    
# ###
# def pool_init(image_base, mask_base, shared_image, shared_mask, image, mask):
#     shared_image = np.ctypeslib.as_array(image_base.get_obj())
#     shared_image = shared_image.reshape(*image.shape)
#     shared_image[:] = image[:]
    
#     shared_mask = np.ctypeslib.as_array(mask_base.get_obj())
#     shared_mask = shared_mask.reshape(*mask.shape)
#     shared_mask[:] = mask[:]



    
    
    
###
def compare_data(*tf_datasets):
    """
    Shows two dataset's image and label.
        
    Parameters
    ----------
    tf_datasets : 2 tf.datasets
    
    """
    
    tfd1, tfd2 = tf_datasets
    
    for e1, e2 in zip(tfd1, tfd2):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        for row_axes, ee in zip(axes, (e1, e2)):
            row_axes[0].imshow(ee[0].numpy().astype(np.float32) / 255.)
            row_axes[1].imshow(ee[1])
            
        
#         for i, ee1, ee2 in zip(range(len(axes[0])), e1, e2):
#             axes[:, i][0].imshow(ee1[0].numpy().astype(np.float32) / 255.)
#             axes[:, i][1].imshow(ee[1])


###
def extract_image_label_only(tfdata, image_feature_description = image_feature_description):
    parsed_example = tf.io.parse_single_example(tfdata, image_feature_description)
    
    h, w, ch = parsed_example['height'], parsed_example['width'], parsed_example['num_channels']
    
    img_dtype = parsed_example['img_dtype']
    image_decoded = tf.io.decode_raw(parsed_example['img_bytes'], out_type = 'uint8')
    image = tf.reshape(image_decoded, (h, w, ch))
    
#     image_float32 = tf.cast(image, dtype = tf.float32)
    
    mask_dtype = parsed_example['mask_dtype']
    mask_decoded = tf.io.decode_raw(parsed_example['mask'], out_type = 'bool')
    mask = tf.reshape(mask_decoded, (h, w, 1))
    
#     mask_uint8 = tf.cast(mask, dtype = tf.uint8)
    
    return image, mask


###
def data_aug(image, label):
    label = tf.cast(label, tf.uint8)
    
    image_shape = tf.cast(tf.shape(image), tf.float32)
    h, w, ch = image_shape[0], image_shape[1], image_shape[2]

    ### random shift
    shift_vector = (
                    h * tf.random.uniform([], -0.05, 0.05), 
                    w * tf.random.uniform([], -0.05, 0.05)
                    )
    image_1 = tfa.image.translate(image, shift_vector)
    label_1 = tfa.image.translate(label, shift_vector)


    ### random flip
    vertial_cond = tf.cast(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1)[0][0], tf.bool)
    image_2 = tf.cond(
                      vertial_cond,
                      lambda : tf.image.flip_left_right(image_1),
                      lambda : image_1
                    )
    label_2 = tf.cond(
                  vertial_cond,
                  lambda : tf.image.flip_left_right(label_1),
                  lambda : label_1
                )

    horizontal_cond = tf.cast(tf.random.categorical(tf.math.log([[0.5, 0.5]]), 1)[0][0], tf.bool)
    image_3 = tf.cond(
                      horizontal_cond,
                      lambda : tf.image.flip_up_down(image_2),
                      lambda : image_2
                    )
    label_3 = tf.cond(
                  horizontal_cond,
                  lambda : tf.image.flip_up_down(label_2),
                  lambda : label_2
                )

    ### random rotation
    rotate_angle = tf.random.uniform([], -45, 45)

    image_4 = tfa.image.rotate(image_3, rotate_angle)
    label_4 = tfa.image.rotate(label_3, rotate_angle)


    ### random shear
    shear_alpha = tf.random.uniform([], -0.27, 0.27) # about [-15 degree ~ +15 degree ]
  
    image_5 = tfa.image.shear_x(image_4, shear_alpha, 255)
    label_5 = tf.image.rgb_to_grayscale(
                                        tfa.image.shear_x(tf.image.grayscale_to_rgb(label_4), shear_alpha, 0)
                                        )
    
    verify_label_values(label_5)
    
#     image_5 = tf.keras.preprocessing.image.random_shear(image_4.numpy(), intensity = shear_angle, row_axis=0, col_axis=1, channel_axis=2)
#     label_5 = tf.keras.preprocessing.image.random_shear(label_4.numpy(), intensity = shear_angle, row_axis=0, col_axis=1, channel_axis=2)
    
#     image_5 = image_4
#     label_5 = label_4
    
    ### random resizing
    resize_factor = tf.random.uniform([], 0.6, 2.0)
    resize_h = tf.cast(tf.math.round(h * resize_factor), tf.int32)
    resize_w = tf.cast(tf.math.round(w * resize_factor), tf.int32)

    image_6 = tf.image.resize(image_5, (resize_h, resize_w))
    label_6 = tf.image.resize(label_5, (resize_h, resize_w), method = "nearest")
#     tf.print(tf.shape(image_6), tf.shape(label_6))
    
    ### center crop
    image_7, label_7 = center_crop(image, label)
    
    return tf.cast(image_7, dtype = tf.float32), label_7


###
@tf.function
def center_crop(image, label):
    ch = tf.cast(tf.shape(image), tf.float32)[2]
    
    ### center crop
    target_size = tf.constant(102)
#     central_fraction = tf.divide(target_size, tf.shape(image_6)[0])
#     image_7 = tf.image.central_crop(image_6, central_fraction)
#     label_7 = tf.image.central_crop(label_6, central_fraction)
    offset_height = tf.subtract(
                                tf.round(tf.divide(tf.shape(image)[0], 2)), tf.round(tf.divide(target_size, 2))
                                )
    offset_width = tf.subtract(
                                tf.round(tf.divide(tf.shape(image)[1], 2)), tf.round(tf.divide(target_size, 2))
                                )
    
    offset_height_c = tf.cast(offset_height, tf.int32)
    offset_width_c = tf.cast(offset_width, tf.int32)
    
    image_cropped = tf.image.crop_to_bounding_box(image, offset_height_c, offset_width_c, target_size, target_size)
    label_cropped = tf.image.crop_to_bounding_box(label, offset_height_c, offset_width_c, target_size, target_size)
    
    tf.assert_equal(tf.shape(image_cropped)[0], target_size, "Augmented data's shape is not [{0}, {0}, {1}]".format(target_size, ch))
    
    return image_cropped, label_cropped


###
def _resize_label_only(image, label):
    ### resizing label to (54, 54)
    label_resized = tf.image.resize(tf.cast(label, dtype = tf.uint8), (54, 54), method = "nearest")
#     label_one_hotted = tf.one_hot(
#                                     tf.squeeze(label_resized, axis = -1), depth = 2
#                                 )
                                
    return image, label_resized

    
### 
def standardize_and_resize_label_only(image, label):
    image_standardized = tf.image.per_image_standardization(tf.cast(image, dtype=tf.float32))
    _, label_resized = _resize_label_only(image, label)
    
    return image_standardized, label_resized
    
###
def val_set_process(image, label):
    ### center crop
    target_size = tf.constant(102)
#     central_fraction = tf.divide(target_size, tf.shape(image_6)[0])
#     image_7 = tf.image.central_crop(image_6, central_fraction)
#     label_7 = tf.image.central_crop(label_6, central_fraction)
    offset_height = tf.subtract(
                                tf.round(tf.divide(tf.shape(image)[0], 2)), tf.round(tf.divide(target_size, 2))
                                )
    offset_width = tf.subtract(
                                tf.round(tf.divide(tf.shape(image)[1], 2)), tf.round(tf.divide(target_size, 2))
                                )
    
    offset_height_c = tf.cast(offset_height, tf.int32)
    offset_width_c = tf.cast(offset_width, tf.int32)
    
    image_1 = tf.image.crop_to_bounding_box(image, offset_height_c, offset_width_c, target_size, target_size)
    label_1 = tf.image.crop_to_bounding_box(label, offset_height_c, offset_width_c, target_size, target_size)  

    return image_1, label_1


###
def construct_tiles_meta(*args, **kwargs):
    return merge_row_meta(*args, **kwargs)

###
def merge_row_meta(csv_root, filter_by_lowband_density = True):
    """
    csv_root : csv root path.
    
    """
    csv_root = Path(csv_root)
    l1 = list(csv_root.glob("*tiles_meta.csv"))
    
    if l1:
        print("tiles_meta.csv was found in csv_root path. {}\nLoading the file instead of making new one...".format(l1[0]))
        return pd.read_csv(l1[0]).iloc[:, 2:]
    
    else:
        tiles_meta = pd.DataFrame(columns = pd.read_csv(next(csv_root.glob("**/*.csv")), index_col = 0).columns)

        pb = csp.Progressbar()
        el = []
        for i, e in enumerate(csv_root.glob("**/*.csv")):
            try:
                tiles_meta = tiles_meta.append(pd.read_csv(e, index_col = 0))
                if filter_by_lowband_density:
                    tiles_meta = tiles_meta.loc[tiles_meta['lowband_density'] > 1000, ] # Exclude non-tissue tiles
                pb.show(i, "Merge all tile meta dataframes... ")
                
            except Exception as exc:
                print("An error occured during processing {}".format(e))
                el.append((e, exc.__class__.__name__, exc))
            
        tiles_meta.to_csv(csv_root / "tiles_meta.csv")
    
    return tiles_meta.reset_index()


###
def make_meta_ready(tiles_meta, random_seed):
    glom_tile_indices = tiles_meta.index[tiles_meta['mask_density'] > 0]
    
    nonglom_indices = tiles_meta.index[tiles_meta['mask_density'] == 0]
    
    np.random.seed(random_seed)
    nonglom_selected_indices = np.random.choice(nonglom_indices, size = len(glom_tile_indices))
    
#     nonglom_indices_r = set(nonglom_indices).difference(nonglom_selected_indices)
    
    tiles_meta = tiles_meta.loc[np.concatenate((glom_tile_indices, nonglom_selected_indices), axis = 0), ].sort_index(axis = 0).reset_index()
    
    for k,v in locals().items():
        globals()[k] = v
    
    return tiles_meta


###
def find_shear_alpha(angle):
    """
    Find alpha value of tfa.image.shear_x in accordance with the shear angle which you want
    
    angle : a radian angle
    """
    return tf.sqrt(
                    1 / tf.square(tf.cos(angle)) - 1
                )


###
@tf.function
def verify_label_values(label):
    """
    Check label values if the unique values are in [0, 1].
    
    """
    y0 = tf.constant(0, dtype = tf.uint8)
    y1 = tf.constant(1, dtype = tf.uint8)
    
    tf.Assert(
        tf.reduce_all(
                        tf.logical_or(tf.equal(label, y0), tf.equal(label, y1))
                    ),
            ["Label array has other values than [0, 1] after sheared."])
    
    
    
###
def crop_and_cast(image, label):
    
    return center_crop(tf.cast(image, dtype = tf.float32), tf.cast(label, dtype = tf.uint8))


###
def train_preprocess(tfrecord):
    image, label = extract_image_label_only(tfrecord)
    image_aug, label_aug = data_aug(image, label)
    image_std, label_std = standardize_and_resize_label_only(image_aug, label_aug)
    
    return image_std, label_std


###
def val_preprocess(tfrecord):
    image, label = extract_image_label_only(tfrecord)
    image_1, label_1 = val_set_process(image, label)
    image_std, label_std = standardize_and_resize_label_only(image_1, label_1)
    
    return image_std, label_std


###
def test_preprocess(tfrecord):
    image, label = extract_image_label_only_for_test(tfrecord)
    image_std, label_std = standardize_and_resize_label_only(image, label)
    
    return image_std, label_std


###
def make_dataset_(shard_index, num_shards = None, num_repeat = None, filepaths = None, preprocess_func = None, batch_size = None):
    filepaths = filepaths.shard(num_shards, shard_index)
    tfrecords_dataset = tf.data.TFRecordDataset(filepaths, compression_type = "GZIP")
    dataset = tfrecords_dataset.repeat(num_repeat).map(preprocess_func).shuffle(100)
    
    return dataset.batch(batch_size)
    

###
def make_dataset_val(shard_index, num_shards = None, num_repeat = None, filepaths = None, preprocess_func = None, batch_size = None):
    filepaths = filepaths.shard(num_shards, shard_index)
    tfrecords_dataset = tf.data.TFRecordDataset(filepaths, compression_type = "GZIP")
    dataset = tfrecords_dataset.map(preprocess_func)
    
    return dataset.batch(batch_size)
    
    
###
def make_dataset_2(shard_index, num_shards = None, num_repeat = None, filepaths = None, preprocess_func = None, batch_size = None):
    filepaths = filepaths.shard(num_shards, shard_index)
    tfrecords_dataset = tf.data.TFRecordDataset(filepaths, compression_type = "GZIP")
    dataset = tfrecords_dataset.repeat(num_repeat).map(preprocess_func).shuffle(100)
    
    return dataset.batch(batch_size)


###
###
def write_tfrecord_tiles_mp_main_test3(row_number, image_index, image_id, output_dir, tile_cols, tile_size, stride):

    dir_path = Path(output_dir) / "row{}".format(row_number)
    dir_path.mkdir(exist_ok = True)
    print(id(image), id(mask))
    
    image_shape = image.shape
    
    ## set tile_df
    tile_df_path = dir_path / "{}_row{}_tile-df.csv".format(image_id, row_number)
    
    if tile_df_path.is_file():
        tile_df = pd.read_csv(tile_df_path, index_col = 0)
    else:
        tile_df = pd.DataFrame(columns = ['img_index', 'img_id', "image_shape", 'tile_id', 'tile_rel_path', 'tile_row_num', 'tile_col_num', 
                                          'lowband_density', 'mask_density', "zero_padded"])
        
    tileID = 0
    
    for col_number in range(tile_cols):
        dataset_file_path = dir_path / 'row{}_col{}.tfrecord'.format(row_number, col_number)
        relative_path = image_id + "/row{0}/row{0}_col{1}.tfrecord".format(row_number, col_number)

        lower_col_range = col_number * stride
        image_tile, orig_image_shape = get_tile(image, tile_size, row_number, col_number, stride)
        tile_mask, _ = get_tile(mask, tile_size, row_number, col_number, stride)
        
        print("image_tile.shape = {}\nimage_id = {}, row_number = {}, col_number = {}".format(image_tile.shape, image_id, row_number, col_number))
        if image_tile.shape != (tile_size, tile_size, 3):
            raise ValueError("tile size is not {} but {}\nimage_id = {}, row_number = {}, col_number = {}".format((tile_size, tile_size, 3), image_tile.shape, image_id, row_number, col_number))
        
        num_records = create_tfrecord(image_index, image_tile, tile_mask, tileID, row_number, col_number, dataset_file_path)

        # populate the metadata for this tile
        img_hist = np.histogram(image_tile)
        lowband_density = np.sum(img_hist[0][0:4])
        mask_density = np.count_nonzero(tile_mask)
        tile_df.loc[tileID, :] = {'img_index':image_index, 'img_id':image_id, "image_shape": image_shape, 'tile_id': tileID,
                                  'tile_rel_path':relative_path, 'tile_col_num':col_number, 'tile_row_num':row_number,'lowband_density':lowband_density, 
                                  'mask_density':mask_density, "zero_padded":orig_image_shape}

        tileID += 1
        
        del image_tile, tile_mask, orig_image_shape
        
        
    tile_df.to_csv(tile_df_path)


###
def extract_image_label_only_for_test(tfdata, image_feature_description = image_feature_description):
    parsed_example = tf.io.parse_single_example(tfdata, image_feature_description)
    
    h, w, ch = parsed_example['height'], parsed_example['width'], parsed_example['num_channels']
    
    img_dtype = parsed_example['img_dtype']
    image_decoded = tf.io.decode_raw(parsed_example['img_bytes'], out_type = 'uint8')
    image = tf.reshape(image_decoded, (h, w, ch))
    
#     image_float32 = tf.cast(image, dtype = tf.float32)
    
    mask_dtype = parsed_example['mask_dtype']
    mask_decoded = tf.io.decode_raw(parsed_example['mask'], out_type = 'bool')
    mask = tf.reshape(mask_decoded, (h, w, 3))
    
#     mask_uint8 = tf.cast(mask, dtype = tf.uint8)
    
    return image, mask


# New version
def rle_encode_less_memory(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


###
def fill_backplate(tile, meta_data, backplate):
    """
    tile : 3 dimensional array
    
    """
    
    row_pos = meta_data['tile_row_num']
    col_pos = meta_data['tile_col_num']
    
    backplate[row_pos*strides : row_pos*strides + orig_tile_size, col_pos*strides : col_pos*strides + orig_tile_size] = tile