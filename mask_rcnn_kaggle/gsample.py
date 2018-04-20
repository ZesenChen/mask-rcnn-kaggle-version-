import cv2
import os
import numpy as np
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.morphology import label
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

def transform(image_path, img_size):
    im_cv = cv2.resize(cv2.imread(image_path),(img_size,img_size))
    im = Image.fromarray(im_cv)
    img1,img2,img3,img4,img5 = np.array(im.transpose(Image.FLIP_LEFT_RIGHT)), \
                               np.array(im.transpose(Image.FLIP_TOP_BOTTOM)), \
                               np.array(im.transpose(Image.ROTATE_90)), \
                               np.array(im.transpose(Image.ROTATE_180)), \
                               np.array(im.transpose(Image.ROTATE_270))
    return [im_cv,img1,img2,img3,img4,img5]

def transform_mask(mask):
    mask0 = np.array(mask)
    mask_num = mask0.shape[2]
    mask1,mask2,mask3,mask4,mask5 = np.zeros(mask0.shape,dtype=np.bool), \
                                    np.zeros(mask0.shape,dtype=np.bool), \
                                    np.zeros(mask0.shape,dtype=np.bool), \
                                    np.zeros(mask0.shape,dtype=np.bool), \
                                    np.zeros(mask0.shape,dtype=np.bool)
    for i in range(mask_num):
        tmp_mask = mask0[:,:,i]
        im = Image.fromarray(tmp_mask)
        mask1[:,:,i],mask2[:,:,i],mask3[:,:,i],mask4[:,:,i],mask5[:,:,i] = \
                               np.array(im.transpose(Image.FLIP_LEFT_RIGHT)), \
                               np.array(im.transpose(Image.FLIP_TOP_BOTTOM)), \
                               np.array(im.transpose(Image.ROTATE_90)), \
                               np.array(im.transpose(Image.ROTATE_180)), \
                               np.array(im.transpose(Image.ROTATE_270))
    return [mask0,mask1,mask2,mask3,mask4,mask5]
   
def make_df(train_path, test_path, img_size):
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]
    X_train = []#np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y_train = []
    #Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    for i, id_ in enumerate(train_ids):
        path = train_path + id_
       # tmp_img = transform(path + '/images/' + id_ + '.png',img_size)
       # for img in tmp_img:
        #    X_train.append(img)
        img = cv2.imread(path + '/images/' + id_ + '.png')
        img = cv2.resize(img, (img_size, img_size))
        X_train.append(img)
        mask_num = len(next(os.walk(path + '/masks/'))[2])
        mask = np.zeros((img_size, img_size, mask_num),dtype=np.bool)
        j = 0
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_size, img_size))
            #mask_ = mask_[:, :, np.newaxis]
            mask[:,:,j] = mask_
            j += 1
            #mask = np.maximum(mask, mask_)
        #tmp_mask = transform_mask(mask)
        #for ms in tmp_mask:
        Y_train.append(mask)
    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
    sizes_test = []
    for i, id_ in enumerate(test_ids):
        path = test_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_size, img_size))
        X_test[i] = img

    return X_train, Y_train, X_test, sizes_test
    

def generator(xtr, xval, ytr, yval, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr, seed=7)
    mask_datagen.fit(ytr, seed=7)
    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)

    train_data = []
    train_target = []
    train_generator = zip(image_generator, mask_generator)
    
    for i in train_generator:
        #print(i[0][batch_size-1].shape)
        for j in range(i[0].shape[0]):
                train_data.append(i[0][j])
                train_target.append(i[1][j])

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval, seed=7)
    mask_datagen_val.fit(yval, seed=7)
    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
    val_generator = zip(image_generator_val, mask_generator_val)
    val_data = []
    val_target = []
    
    for k in val_generator:
        #print(k[0][batch_size-1].shape)
        for j in range(k[0].shape[0]):
                val_data.append(k[0][j])
                val_target.append(k[1][j])
    
    return train_data, train_target, val_data, val_target
