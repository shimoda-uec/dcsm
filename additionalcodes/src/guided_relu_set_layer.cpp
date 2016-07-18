#include <algorithm>
#include <vector>

#include "caffe/layers/guided_relu_set_layer.hpp"

namespace caffe {

template <typename Dtype>
void GuidedReLUSetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}
 
template <typename Dtype>
void GuidedReLUSetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();//derivative
  const Dtype* bottom_data2 = bottom[1]->cpu_data();//conv value
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int inner_num=bottom[1]->channels()*bottom[1]->height()*bottom[1]->width();
  for (int n = 0; n < 1; ++n) {
  	for (int e=0;e<bottom[0]->num();++e){
	     for(int c=0;c<inner_num;++c){
        	top_data[e*inner_num+c] = bottom_data[e*inner_num+c]*(bottom_data2[n*inner_num+c] > 0)*(bottom_data[e*inner_num+c]>0);
              }
  	}
  }
}

template <typename Dtype>
void GuidedReLUSetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}
#ifdef CPU_ONLY
STUB_GPU(GuidedReLUSetLayer);
#endif

INSTANTIATE_CLASS(GuidedReLUSetLayer);
REGISTER_LAYER_CLASS(GuidedReLUSet);
}  // namespace caffe
