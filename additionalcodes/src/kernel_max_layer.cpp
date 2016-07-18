#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/kernel_max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}

template <typename Dtype>
void KernelMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();//derivative
  //top[0]->Reshape(1,1,1,1);
  Dtype* top_data = top[0]->mutable_cpu_data();
  //Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  Dtype mval;
  int mid ;
  int index ;
  int tindex ;
  for (int n = 0; n < bottom[0]->num(); ++n) {
  	for (int i=0;i<bottom[0]->height();++i){
       for (int j = 0; j < bottom[0]->width(); ++j) {
          mval=-FLT_MAX;
          mid =-1;
       	  for (int c=0;c <bottom[0]->channels();++c){
       	     index =  c*bottom[0]->height()*bottom[0]->width() + bottom[0]->width()*i + j;
       	     tindex =  bottom[0]->width()*i + j;
       	  	if(mval<bottom_data[index]){
       	  		mval=bottom_data[index];
       	  		mid=tindex;
       	  	}
       	  }
       	  top_data[mid]=mval;
       }
  	}
  	//top_data[0]=en;
    bottom_data += bottom[0]->offset(1, 0);
    top_data+= top[0]->offset(1, 0);
  }
}

template <typename Dtype>
void KernelMaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(KernelMaxLayer);
#endif

INSTANTIATE_CLASS(KernelMaxLayer);
REGISTER_LAYER_CLASS(KernelMax);
}  // namespace caffe
