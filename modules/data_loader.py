# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:26:56 2020

@author: yari
"""

from torch.utils import data as D
from os.path import join, abspath #,split, abspath, splitext, split, isdir, isfile
import numpy as np
import cv2
import os

import pandas as pd

import glob

from scipy.io import loadmat

import pywt
import math # added by @dv
#S3
import boto3
from PIL import Image
from io import BytesIO
from botocore.client import Config # added by @dv
from msnet.msnet_parts import crop as crop_img #@dv
from msnet.msnet_parts import make_bilinear_weights #@dv
import torch.nn.functional as F #@dv
import torch #@dv
import matplotlib.pyplot as plt

#### by @dv for swt
class Dataset_s3_swt(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    # def __init__(self, bucket, keys, s3Get, prepare=False, transform=None): # original commented by @dv
    def __init__(self, bucket, keys, s3Get, prepare=True, transform=None): # @dv
        self.df=keys
        self.bucket=bucket
        self.s3Get=s3Get
        self.transform=transform
        self.prepare=prepare

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        key=self.df[index]
        img= self.s3Get(self.bucket, key)
        if '.tiff' in key:
            key = key.replace('.tiff','.png') # by @dv for layer_binary files
        ctour=self.s3Get(self.bucket, key.replace('data', 'layer_binary'))

        ctour=np.array(ctour, dtype=np.float32)
        img=np.array(img, dtype=np.float32)

        # #### to process tiff files @dv #####
        # if self.prepare: 
        #     img, ctour= prepare_img_mat(img), prepare_ctour(ctour)

        # _, h, w = img.shape
        # if (h%2) != 0:
        #     img=img[:,:-1,:]
        #     ctour=ctour[:,:-1,:]
        # coeffs = pywt.swt2(img,'haar',level=1,start_level=0,trim_approx=True)
        # img = coeffs[0] ### passing the approximate coeff in the input

        #### to process regular files @dv #####
        h, w = img.shape
        if (h%2) != 0:
            img=img[:-1]
            ctour=ctour[:-1]
        coeffs = pywt.swt2(img,'haar',level=1,start_level=0,trim_approx=True)
        # img = coeffs[0] ### passing the approximate coeff in the input
        # img = np.stack((coeffs[1][0],coeffs[1][1],coeffs[1][2])) ### passing the detail coeff in the input
        # img = np.concatenate((coeffs[1][0],coeffs[1][1],coeffs[1][2]),axis=1) ### passing the detail coeff in the input
        img = pywt.iswt2(coeffs, 'haar')
        
        if self.transform:
            img=self.transform(img)
            ctour=self.transform(ctour)
        
        if self.prepare: ## original, commented by @dv
            img, ctour= prepare_img(img), prepare_ctour(ctour)
            # img, ctour= prepare_img_mat(img), prepare_ctour(ctour)
            # img, ctour= prepare_img_mat_2(img), prepare_ctour(ctour)

        (data_id, _) = os.path.splitext(os.path.basename(key))



        return {'image': img, 'mask' : ctour , 'id': data_id}



class Dataset_s3(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    # def __init__(self, bucket, keys, s3Get, prepare=False, transform=None): # original commented by @dv
    def __init__(self, bucket, keys, s3Get, prepare=True, transform=None): # @dv
        self.df=keys
        self.bucket=bucket
        self.s3Get=s3Get
        self.transform=transform
        self.prepare=prepare

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        key=self.df[index]
        img= self.s3Get(self.bucket, key)
        if '.tiff' in key:
            key = key.replace('.tiff','.png') # by @dv for layer_binary files
            
        ctour=self.s3Get(self.bucket, key.replace('data', 'layer_binary'))

        ctour=np.array(ctour, dtype=np.float32)
        img=np.array(img, dtype=np.float32)

        # ### following lines added as replacement to swt method
        # img_H, img_W = img.shape
        # w=get_wt_dv(img,'haar', mode='periodic', level=1) ## by @dv
        # img = w['cA1']
        # img = np.expand_dims(img,0)
        # weight_deconv2 =  make_bilinear_weights(4, 1)
        # upsample2 = F.conv_transpose2d(torch.from_numpy(img), weight_deconv2, stride=2)
        # img = np.array(crop_img(upsample2, img_H, img_W)[0][0])
        # #####
        
        if self.transform:
            img=self.transform(img)
            ctour=self.transform(ctour)

        if self.prepare: ## original, commented by @dv
            # img, ctour= prepare_img(img), prepare_ctour(ctour) ## original, commented by @dv
            img, ctour= prepare_img_mat(img), prepare_ctour(ctour) ## added for tiff processing by @dv
        (data_id, _) = os.path.splitext(os.path.basename(key))



        return {'image': img, 'mask' : ctour , 'id': data_id}

class Dataset_s3_w(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    def __init__(self, bucket, keys, s3Get,  transform=None):
        self.df=keys
        self.bucket=bucket
        self.s3Get=s3Get
        self.transform=transform
       # self.prepare=prepare

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        key=self.df[index]
        key = key.replace("\\","/") ## temporary solution for testing L2 through EC2
        img= self.s3Get(self.bucket, key)
        if '.tiff' in key:
            key = key.replace('.tiff','.png') # by @dv for layer_binary files
        # ctour=self.s3Get(self.bucket, key.replace('data', 'layer_binary')) # removing ctour temporarily for prediction

        # ctour=np.array(ctour, dtype=np.float32) # removing ctour temporarily for prediction
        img=np.array(img, dtype=np.float32)

        if self.transform:
            img=self.transform(img)
            # ctour=self.transform(ctour) # removing ctour temporarily for prediction

        #w=pywt.wavedec2(prepare_w(img),'haar', mode='periodic', level=4)
        # w=get_wt(img,'haar', mode='periodic', level=4) ## original, commented by @dv
        w=get_wt_dv(img,'dmey', mode='periodic', level=4) ## by @dv

        # img, ctour= prepare_img(img), prepare_ctour(ctour) # commented by @dv
        # img, ctour= prepare_img_mat(img), prepare_ctour(ctour)
        img = prepare_img_mat(img) # removing ctour temporarily for prediction


        (data_id, _) = os.path.splitext(os.path.basename(key))
    
        # return {'image': img, 'mask' : ctour , 'id': data_id, 'w': w}
        return {'image': img, 'id': data_id, 'w': w} # removing ctour temporarily for prediction

### @dv: added from Dr. Yari's Multiscale GitHub code
class BasicDataset(D.Dataset):
    """
    dataset from directory
    returns img after preperation, no label
    """
    def __init__(self, root, ext):
        self.root=root
        self.ext=ext
        self.rel_paths=glob.glob(join(root, '*.{}'.format(ext)))

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, index):

        # get image
        img_abspath= abspath(self.rel_paths[index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

        img=cv2.imread(img_abspath) ### original, commented by @dv

        img= prepare_img_3c(img) ### original, commented by @dv

        # img = Image.open(img_abspath) ## added by @dv
        # img=np.array(img, dtype=np.float32) ## added by @dv
        
        (data_id, _) = os.path.splitext(os.path.basename(img_abspath))


        return {'image': img, 'id':data_id}

class S3ImagesInvalidExtension(Exception):
    pass

class S3ImagesUploadFailed(Exception):
    pass

class S3Images(object):

    """Useage:

        images = S3Images(aws_access_key_id='fjrn4uun-my-access-key-589gnmrn90',
                          aws_secret_access_key='4f4nvu5tvnd-my-secret-access-key-rjfjnubu34un4tu4',
                          region_name='eu-west-1')
        im = images.from_s3('my-example-bucket-9933668', 'pythonlogo.png')
        im
        images.to_s3(im, 'my-example-bucket-9933668', 'pythonlogo2.png')
    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name=None):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key,
                                     region_name=region_name, config=Config(connect_timeout=5)) ### config added by @dv


    def from_s3(self, bucket, key):
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return Image.open(obj["Body"])


    def to_s3(self, img, bucket, key):
        buffer = BytesIO()
        img.save(buffer, self.__get_safe_ext(key))
        buffer.seek(0)
        sent_data = self.s3.put_object(Bucket=bucket, Key=key, Body=buffer)
        if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))

    def __get_safe_ext(self, key):
        ext = os.path.splitext(key)[-1].strip('.').upper()
        if ext in ['JPG', 'JPEG']:
            return 'JPEG'
        elif ext in ['PNG']:
            return 'PNG'
        else:
            raise S3ImagesInvalidExtension('Extension is invalid')




def prepare_img_mat(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255
        img=np.expand_dims(img, axis=2)
        img=np.repeat(img,3,axis=2)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img


"""assuming img is made up of 3 channels"""
def prepare_img_mat_2(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255
        img=img.transpose(1,2,0)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img
    
def prepare_img_mat_tiff(img):
        img=np.array(img, dtype=np.float32)
        #img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=img*255

        img -= np.array((104.00698793,116.66876762,122.67891434))

        return img

def prepare_img(img):
        img=np.expand_dims(img,axis=2)
        img=np.repeat(img,3,axis=2)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img

def prepare_ctour(ctour):
        #ctour=np.array(ctour, dtype=np.float32)
        ctour = (ctour > 0 ).astype(np.float32)
        ctour=np.expand_dims(ctour,axis=0)
        return ctour

def prepare_img_3c(img):
        img=np.array(img, dtype=np.float32)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img

def prepare_w(img):
        img=np.array(img, dtype=np.float32)
        img=np.expand_dims(img,axis=0)
        return img


def wt_scale(wt):
    return 255*(wt-np.min(wt))/(np.max(wt)-np.min(wt))

def get_wt(im, wname, mode, level,scaleit=False):
    w=pywt.wavedec2(im,wname, mode=mode, level=level)
    if scaleit:
        wt={f'cA{level}': prepare_w(wt_scale(w[0]))}
        for i in range(1,level):
            wt.update({f'cH{i}': prepare_w(wt_scale(w[-i][0]))})
            wt.update({f'cV{i}': prepare_w(wt_scale(w[-i][1]))})
            wt.update({f'cD{i}': prepare_w(wt_scale(w[-i][2]))})
    else:
        wt={f'cA{level}': prepare_w(w[0])}
        # for i in range(1,level): # dr.yari
        for i in range(1,level+1): # @dv
            wt.update({f'cH{i}': prepare_w(w[-i][0])})
            wt.update({f'cV{i}': prepare_w(w[-i][1])})
            wt.update({f'cD{i}': prepare_w(w[-i][2])})
    return wt

### added by @dv from another file. originally by Dr. Yari
def crop(variable, th, tw):
        # h, w = variable.shape[2], variable.shape[3] ## original, commented by @dv
        h, w = variable.shape[1], variable.shape[2] ## added by @dv
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        # return variable[:, :, y1 : y1 + th, x1 : x1 + tw] ## original, commented by @dv
        return variable[:, y1 : y1 + th, x1 : x1 + tw] ## by @dv
    
    
### added by @dv. Crop out wavelet transforms so that they do not create a concatenation issue
### especially for non-haar wavelets ## @dv
def get_wt_dv(im, wname, mode, level,scaleit=False):
    rows, cols = im.shape
    w=pywt.wavedec2(im,wname, mode=mode, level=level)
    wt={f'cA{level}': prepare_w(w[0])}
    for i in range(1,level+1):
        w_row = math.ceil(rows/(2**i))
        w_col = math.ceil(cols/(2**i))
        
        wt.update({f'cH{i}': crop(prepare_w(w[-i][0]),w_row,w_col)})
        wt.update({f'cV{i}': crop(prepare_w(w[-i][1]),w_row,w_col)})
        wt.update({f'cD{i}': crop(prepare_w(w[-i][2]),w_row,w_col)})
    return wt
# def make_wlets(original, wname='db1', level=4):
#    # original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#     original = np.array(original, dtype=np.float32)
#     coeffs=pywt.wavedec2(original, wname, level=level)
#     #LL, (LH, HL, HH) = pywt.dwt2(original, 'db1')
#     cA4, (cH4,cV4,cD4),(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)=coeffs
#     return cH4,cV4,cD4, cH3,cV3,cD3,cH2,cV2,cD2, cH1,cV1,cD1

# enum ImreadModes
# {
#     IMREAD_UNCHANGED           = -1,
#     IMREAD_GRAYSCALE           = 0,
#     IMREAD_COLOR               = 1,
#     IMREAD_ANYDEPTH            = 2,
#     IMREAD_ANYCOLOR            = 4,
#     IMREAD_LOAD_GDAL           = 8,
#     IMREAD_REDUCED_GRAYSCALE_2 = 16,
#     IMREAD_REDUCED_COLOR_2     = 17,
#     IMREAD_REDUCED_GRAYSCALE_4 = 32,
#     IMREAD_REDUCED_COLOR_4     = 33,
#     IMREAD_REDUCED_GRAYSCALE_8 = 64,
#     IMREAD_REDUCED_COLOR_8     = 65,
#     IMREAD_IGNORE_ORIENTATION  = 128,}
