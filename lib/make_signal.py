import sys
sys.path.append('./caffe/python')#set your own caffe route
import caffe
import numpy as np

class MakeSignalLayer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        num,ch,height,width=bottom[1].data.shape
        top[0].reshape(3,ch,height,width)

    def forward(self, bottom, top):
        num,ch,height,width=bottom[1].data.shape
        for n in range(3):
           for c in range(ch):
              if(bottom[0].data[0,n,0,0]==c):
                 top[0].data[n,c,:,:]=1
              else:
                 top[0].data[n,c,:,:]=0

    def backward(self, top, propagate_down, bottom):
        pass
