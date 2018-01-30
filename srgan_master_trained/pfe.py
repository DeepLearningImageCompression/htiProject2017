#! /usr/bin/python
# -*- coding: utf8 -*-

#from srgan_master_trained.main import evaluate

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

import os, sys
import PIL
from PIL import Image

# images_test = os.listdir('D:\datasets\im_test_lr')
# for imageName in images_test:


def evaluate(road_lr, road_hr, imageName):
    ## create folders to save result images
    save_dir = ".\\results\{}".format('evaluate')
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    imageNAME = imageName
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    #valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    #valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    #imid = 64 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    #valid_lr_img = valid_lr_imgs[imid]
    #valid_hr_img = valid_hr_imgs[imid]

    #net_g = SRGAN_g(imageName, t_image, is_train=False, reuse=False)

    imageName = os.path.splitext(imageName)[0]
    imageName = imageName + '.png'
    valid_lr_img = get_imgs_fn(imageName, road_lr)  # if you want to test your own image
    valid_hr_img = get_imgs_fn(imageName, road_hr)  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})

    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir+'/'+imageName+'_gen.jpg')
    tl.vis.save_image(valid_lr_img, save_dir+'/'+imageName+'_lr.jpg')
    tl.vis.save_image(valid_hr_img, save_dir+'/'+imageName+'_hr.jpg')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir+'/'+imageName+'_bicubic.png')





#road_lr = "D:\datasets\im_test_lr\\"
#road_hr = "D:\datasets\im_test_hr\\"

road_lr = "./samples/evaluate/test/lr/"
road_hr = "./samples/evaluate/test/hr/"

images = os.listdir('./samples/evaluate/test/lr/')

for imageName in images:
    evaluate(road_lr, road_hr, imageName)
