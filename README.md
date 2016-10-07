#DCSM:Distinct class-specific saliency maps
By Watal Shimoda, Keiji Yanai.
![Flow](https://github.com/shimoda-uec/dcrm/blob/master/process.png "flow")
##Description
This repository contains the codes for the "DCSM" weakly supervised semantic segmentation method.  
It has been published at ECCV2016.  
Our codes are based on the Caffe deep learning library.  
Note that in the paper we tested with MatConvNet implementation.
You can also get that from here [MatConvNet implementaion](https://github.com/shimoda-uec/mat_dcsm).
Caluculation detail and computational cost are different.
##Requirements
Requirements for Caffe and pycaffe (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/))  
##Install
First, you should clone the repository as below.  
```
git clone https://github.com/shimoda-uec/dcsm.git
```

You need to compile the modified Caffe library.  
If you compile Caffe in own repository, you should locate our additional codes to your own Caffe library precisely.  
See additionalcodes/README.md.  
##Run the demo
See test/README.md.  
##License and Citation
Please cite our paper if it helps your research:
```
@inproceedings{shimodaECCV16  
  Author = {Shimoda, Wataru and Yanai, Keiji},  
  Title = {Distinct class-specific saliency maps},  
  Booktitle = {International Conference on Computer Vision ({ECCV})},  
  Year = {2016}  
}  
```
