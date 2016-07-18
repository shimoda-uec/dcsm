import sys
sys.path.append('./caffe/python')#set your caffe/python route
import os
import caffe
from caffe import layers as L
from caffe import params as P


def conv_setn(bottom,name, nout, ks = 3, stride=1, pad = 0, dilation=1,learn = True):
    paramn=[dict(name=name)]
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad,dilation=dilation,param=paramn)
    bn = L.ReLU(conv, in_place=True)
    return conv, bn
def dcsm_setn(prev,bs):
    km = L.KernelMax(prev)
    sub = L.SubFix(km)
    mn1 = L.MaxNormalizeFix(sub,prior=3)
    tanh1 = L.TanH(mn1)
    subs = L.SubStackFix(tanh1,sweepern=3)
    mn2 = L.MaxNormalizeFix(subs,prior=3)
    tanh2 = L.TanH(mn2)
    bl = L.Bl(tanh2,newsize=bs)
    return km,sub,mn1,tanh1,subs,mn2,tanh2,bl
def dconv_setn(prev,conv,name,nout, ks = 3, stride=1, pad = 0, dilation=1, learn = True):
    paramn=[dict(name=name)]
    relu = L.GuidedReLUSet(prev,conv)
    conv = L.Deconvolution(relu, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad,dilation=dilation,param=paramn,bias_term=False)
    return relu,conv



def vgg_net(total_depth, lmdb, num_classes = 20, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        16:([2, 2, 3, 3,3], "standard"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)
    nunits_list, unit_type = net_defs[total_depth] # nunits_list a list of integers indicating the number of layers in each depth.
    nouts = [64, 128, 256, 512,512] # same for all nets
    pmflag=[0,0,1,1,1]
    # setup the first couple of layers
    #lr=False
    lr=True
    n = caffe.NetSpec()
    n.data = L.Input()
    # make the convolutional body
    iii=0
    ppid='data'
    for nout, nunits in zip(nouts, nunits_list): # for each depth and nunits
        iii=iii+1
        for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
            convn='conv'+str(iii)+'_'+str(unit)
            relun='relu'+str(iii)+'_'+str(unit)
            pn='conv'+str(iii)+'_'+str(unit)+'w'
            print convn
            n[convn], n[relun] = conv_setn(n[ppid],pn, ks = 3, stride = 1, nout = nout, pad = 1,learn=lr)
            pid=id
            ppid=relun
        pooln='pool'+str(iii)
        n[pooln]=L.Pooling(n[ppid], stride = 2, kernel_size = 2,pool=0)
        ppid=pooln

    n['fc6of'], n['relu6'] = conv_setn(n[ppid],'conv6w', ks = 7, stride = 1, nout = 4096, pad =0,dilation=1,learn=lr)
    n['fc7of'], n['relu7'] = conv_setn(n['relu6'],'conv7w', ks = 1, stride = 1, nout = 4096, pad = 0,learn=lr)
    n['fc8']=L.Convolution(n['relu7'], kernel_size=1,num_output=20,param=[dict(name='conv8w')])

    n['gp']=L.Pooling(n['fc8'], pool=0,global_pooling=True)
    n['sortid']=L.SortCh(n['gp'], topk=3)
    n['sweeper'],n['sweepern'],n['sweepeid']=L.AddSweeperFix(n['sortid'] ,sweepern=3,ntop=3)
    n['signal']=L.MakeSignalFix(n['fc8'],n['sweeper'],n['sweepern'])
    n['d7'] = L.Deconvolution(n['signal'], kernel_size=1,num_output=4096,bias_term=False,param=[dict(name='conv8w')])
    n['d7r'],n['d6'] = dconv_setn(n['d7'],n['fc7of'],'conv7w',4096,ks = 1,stride = 1)
    n['d6r'],n['d5'] = dconv_setn(n['d6'],n['fc6of'],'conv6w',512,ks = 7, stride = 1,  pad =0,dilation=1)

    dnouts = [512,512,256,128,64] # same for all nets
    dnouts2 = [512,256,128,64,0] # same for all nets
    dnunits = [3, 3, 3, 0,0] # same for all nets
    end=9
    iii=6
    ppid='d5'
    pnout=512
    alln=1
    bs=512
    for dnout, dnunits in zip(dnouts, dnunits): # for each depth and nunits
        iii=iii-1
        if dnunits>0:
           pooln='unpool_'+str(iii)
           poolmn='pool_'+str(iii)+'m'
           convn='conv'+str(iii)+'_3'
           n[pooln]=L.UnpoolingNomask(n[ppid],n[convn], stride = 2, kernel_size = 2)
           ppid=pooln
        for dunit in range(1, dnunits + 1): # for each unit. Enumerate from 1.
            dunit=3+1-dunit
            if dunit==1:
               dnout=dnouts2[5-iii]
            convn='conv'+str(iii)+'_'+str(dunit)
            drelun='grelu'+str(iii)+'_'+str(dunit)
            dconvn='dconv'+str(iii)+'_'+str(dunit)
            km=dconvn+'km'
            sub=dconvn+'sub'
            mn1=dconvn+'mn1'
            tanh1=dconvn+'tanh1'
            subs=dconvn+'substack'
            mn2=dconvn+'mn2'
            tanh2=dconvn+'tanh2'
            bl=dconvn+'bl'
            pn='conv'+str(iii)+'_'+str(dunit)+'w'
            #bnn='bn'+str(iii)+'_'+str(unit)
            print dconvn
            n[drelun],n[dconvn] = dconv_setn(n[ppid],n[convn],pn,  nout = dnout,ks = 3, stride = 1, pad = 1,dilation=1)
            n[km],n[sub],n[mn1],n[tanh1],n[subs],n[mn2],n[tanh2],n[bl]= dcsm_setn(n[dconvn],bs)
            pid=id
            ppid=dconvn
            pnout=dnout
            alln=alln+1
    n['dcsm'] = L.Eltwise(n['dconv5_3bl'],n['dconv5_2bl'],n['dconv5_1bl'],n['dconv4_3bl'],n['dconv4_2bl'],n['dconv4_1bl'],n['dconv3_3bl'],n['dconv3_2bl'],n['dconv3_1bl'])
    #n['restore'] = L.RestoreForDcsm(n['dcsm'],n['objn'],n['sweepern'],n['overlapid'],shape=bs,bn=bnn)
    #n['dcsmmask'] = L.ArgmaxForDcsmDsize(n['restore'],n['objn'],n['objid'],size=bs,batchsize=bn)
    return n.to_proto()

with open('./gdep.prototxt', 'w') as f:
    #vgg_net(16,'test')
    f.write(str(vgg_net(16,'test')))
