#include <algorithm>
#include <vector>

#include "caffe/layers/make_signal_fix_layer.hpp"

namespace caffe {

template <typename Dtype>
void MakeSignalFixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AddSweeperParameter sweeper_param = this->layer_param_.add_sweeper_param();
  sweepern_=sweeper_param.sweepern();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(sweepern_, channels_,height_,width_);
}
template <typename Dtype>
void MakeSignalFixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* label = bottom[1]->cpu_data();//label from 0
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < 1; ++n) {
      for (int c = 0; c < channels_; ++c) {
      	if (label[c]==1){
    		  for(int i=0;i<channels_; ++i){
        	      	if (i==c){
        		  		caffe_set(height_*width_, Dtype(1.0), top_data);
        	      	}
        	      	else{
        		  		caffe_set(height_*width_, Dtype(0), top_data);
        	      	}
                  top_data += top[0]->offset(0, 1);
               }
        }
      }
      label += bottom[1]->offset(1, 0);
  }

}

template <typename Dtype>
void MakeSignalFixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}



INSTANTIATE_CLASS(MakeSignalFixLayer);
REGISTER_LAYER_CLASS(MakeSignalFix);
}  // namespace caffe
