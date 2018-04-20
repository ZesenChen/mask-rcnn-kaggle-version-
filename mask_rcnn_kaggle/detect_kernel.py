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

from config import Config
import utils
import model as modellib
import visualize
from model import log
from gsample_v0 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_size = 512
batch_size = 1
train_path = '/data_1/chenzesen/kaggle_data/newtrain/'
test_path = '/data_1/chenzesen/2018_bowl/data/data/__download__/stage2_test/'
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

def commask(mask,image_size):
    print('mask shape:',mask.shape)
    mask_num = mask.shape[-1]
    final_mask = np.zeros((image_size,image_size,1))
    for i in range(mask_num):
        tmp_mask = mask[:,:,i]
        tmp_mask = tmp_mask[:,:,np.newaxis]
        final_mask = np.maximum(final_mask,tmp_mask)
    return final_mask

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

class ShapesConfig(Config):
    NAME = "shapes"
    STEPS_PER_EPOCH = 500
    GPU_COUNT = 1
    #MASK_SHAPE = [256,256]
    IMAGES_PER_GPU = 2
    USE_MINI_MASK = True
    NUM_CLASSES = 1 + 1  # background + 1 shapes
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    TRAIN_ROIS_PER_IMAGE = 600
    STEPS_PER_EPOCH =1000
    VALIDATION_STEPS = 5
    MAX_GT_INSTANCES =256
    DETECTION_MAX_INSTANCES = 512
    BACKBONE = "resnet50"

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#creat a new mask-rcnn object
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#load the pretrained mask-rcnn model
model_path = '/data_1/chenzesen/kaggle_data/2018_bowl/logs/mask_rcnn_kernel_tmd3.h5'
model.load_weights(model_path, by_name=True)
X_test, sizes_test, test_ids = make_df(train_path, test_path, img_size)

#detect the kernel and get the masks
preds_test = []
fuck = 0
fuck_id = []
for i in range(X_test.shape[0]):
    print('Detecting the ',i,'th image......')
    results = model.detect([X_test[i]], verbose=1)
    r = results[0]
    if r['masks'].shape[0] != 256 or r['masks'].shape[1] != 256:
        print(r['masks'].shape)
        preds_test.append(np.zeros((1,1,1)))
        fuck_id.append(test_ids[i])
        continue
    preds_test.append(commask(r['masks'],256))
#resize the mask to original
print(fuck_id)
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(cv2.resize(preds_test[i], 
                                (sizes_test[i][1], sizes_test[i][0])))
test_ids = next(os.walk(test_path))[1]
new_test_ids = []
rles = []

#run length enconding
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('2018data_sub.csv', index=False)
