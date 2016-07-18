#! /usr/bin/python
# file: import-caffe-.py
# brief: Caffe importer for DagNN
# author: Karel Lenc and Andrea Vedaldi

# Requires Google Protobuf for Python and SciPy

import sys
sys.path.append('./caffe/python')#set your caffe/python route
import os
import argparse
import code
import re
import numpy as np
from math import floor, ceil
import numpy
from numpy import array
import scipy
import scipy.io
import scipy.misc
import google.protobuf
from ast import literal_eval as make_tuple
from PIL import Image
import caffe
cm = './mp_iter_60000.caffemodel'
proto='./gdep.prototxt'
caffe.set_mode_gpu()

bsize=512
mean = np.zeros((3,int(bsize),int(bsize)))
mean[0,:,:]=104.00699
mean[1,:,:]=116.66877
mean[2,:,:]=122.67892
channel_swap = [2,1,0]
raw_scale=255.0;
center_only=False
input_scale=None
image_dims=[bsize,bsize]
ims=(int(bsize),int(bsize))
data = caffe.Classifier(proto,cm, image_dims=image_dims,mean=mean,
            input_scale=input_scale,
             raw_scale=raw_scale,
            channel_swap=channel_swap)

im = [caffe.io.load_image('./img.jpg')]
im2=[caffe.io.resize_image(im[0], ims)]
im3 = np.zeros((1,ims[0],ims[1],im2[0].shape[2]),dtype=np.float32)
im3[0]=im2[0]
caffe_in = np.zeros(np.array(im3.shape)[[0, 3, 1, 2]],dtype=np.float32)
caffe_in[0]=data.transformer.preprocess('data', im3[0])
#labelsk=np.zeros((1,20,1,1),dtype=np.float32)
#labelsk[0,1,0,0]=1
#labelsk[0,15,0,0]=1
#out = data.forward_all(**{'data': caffe_in,'label':labelsk})
out = data.forward_all(**{'data': caffe_in})
sal=data.blobs['dcrmn'].data
#print sal.shape
rsize=512
zeronp=np.zeros((rsize,rsize),dtype=np.float32)
for i in range(sal.shape[0]):
		salkp=sal[i,0,:,:]
		salk=np.fmax(salkp,zeronp)
		salk2=np.zeros((rsize,rsize,3),dtype=np.float32)
		salk2[:,:,0]=salk
		salk2[:,:,1]=salk
		salk2[:,:,2]=salk
		salk3=salk2*255
		salint = salk3.astype(np.uint8)
		pil_img = Image.fromarray(salint)
		path='./'
		sn=path+'/'+str(i)+'dcrm.jpg'
		pil_img.save(sn)


