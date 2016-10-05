#! /usr/bin/python
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
out = data.forward_all(**{'data': caffe_in})
map=data.blobs['dcsmn'].data
id=data.blobs['sortid'].data
rsize=512
zeronp=np.zeros((rsize,rsize),dtype=np.float32)
ctg=['plane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dog','dtable','horse','moterbike','person','plant','sheep','sofa','train','tv']
for i in range(map.shape[0]):
		mapk=map[i,0,:,:]
		mapk=np.fmax(mapk,zeronp)
		mapk2=np.zeros((rsize,rsize,3),dtype=np.float32)
		mapk2[:,:,0]=mapk
		mapk2[:,:,1]=mapk
		mapk2[:,:,2]=mapk
		mapk3=mapk2*255
		mapint = mapk3.astype(np.uint8)
		pil_img = Image.fromarray(mapint)
		path='./'
		sn=path+'/'+ctg[int(id[0,i,0,0])]+'_rank'+str(i)+'_dcsm.jpg'
		pil_img.save(sn)


