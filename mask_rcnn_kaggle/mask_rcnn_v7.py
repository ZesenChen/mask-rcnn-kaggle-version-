#coding:utf-8
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from config import Config
import utils
import model as modellib
import visualize
from model import log
import skimage
from data import *
from gsample import *
import scipy.io as sio
from copy import copy

#选择GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = '/data_1/chenzesen/kaggle_data/2018_bowl/'

#模型保存和加载的路径
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class shapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes
    #MINI_MASK_SHAPE = (256,256)
    USE_MINI_MASK = False
    TRAIN_ROIS_PER_IMAGE = 600
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 5    
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 512
    BACKBONE = "resnet50"

class InferenceConfig(shapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class shapesDataset(utils.Dataset):
    def __init__(self, image, mask, num):
        self.image = image
        self.mask = mask
        self.count = num
        #self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.height = 512
        self.width = 512

    def load_kernel(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.height,self.width = height,width
        self.add_class("shapes", 1, "kernel")

        for i in range(count):
            #bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height)

    def load_image(self, image_id):
        return np.array(self.image[image_id])

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        mask_num = self.mask[image_id].shape[2]
        return np.array(self.mask[image_id]),np.ones((mask_num,)) #.reshape((self.height,self.width,1)), np.array([1])

if __name__=='__main__':
    img_size = 512
    batch_size = 8
    train_path = '/data_1/chenzesen/kaggle_data/stage1_train/'
    test_path = '/data_1/chenzesen/kaggle_data/stage1_test/'
    X_train, Y_train, X_test, sizes_test = make_df(train_path, test_path, img_size)
    
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=2018)
    config = shapesConfig()
    config.display()

    data_train = shapesDataset(xtr,ytr,len(xtr))
    data_train.load_kernel(len(xtr),img_size,img_size)
    data_train.prepare()
    data_val = shapesDataset(xval,yval,len(xval))
    data_val.load_kernel(len(xval),img_size,img_size)
    data_val.prepare()

    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

    #初始化模型选择
    init_with = "imagenet"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(ROOT_DIR+'logs/mask_rcnn_coco.h5', by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        #用你之前保存的模型继续训练
        model.load_weights('/data_1/chenzesen/kaggle_data/2018_bowl/logs/mask_rcnn_kernel_tmd3.h5', by_name=True)
    #训练heads层参数
    model.train(data_train, data_val,
                learning_rate=config.LEARNING_RATE,
                epochs=90,
                layers="heads")
    #训练全部网络参数
    model.train(data_train, data_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=80,
                layers="all")
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_kernel_wocaonimalegebi.h5")
    model.keras_model.save_weights(model_path)
    print("model weights saved!")
