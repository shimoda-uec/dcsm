#include <cfloat>
#include <vector>

#include "caffe/layers/sub_stack_fix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SubStackFixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AddSweeperParameter sweeper_param = this->layer_param_.add_sweeper_param();
  sweepern_=sweeper_param.sweepern();
}

template <typename Dtype>
void SubStackFixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(sweepern_,1,bottom[0]->height(),bottom[0]->width());
}

template <typename Dtype>
void SubStackFixLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int inner_num=bottom[0]->height()*bottom[0]->width()*bottom[0]->channels();
  for (int n = 0; n < 1; ++n) {
  	for (int e1=0;e1<sweepern_;++e1){
          caffe_set(inner_num, Dtype(0), top_data);
  	   for (int e2=0;e2<sweepern_;++e2){
	       if(e1!=e2){
		  	    caffe_axpy(inner_num, Dtype(1.0), bottom_data , top_data);
	        }
           bottom_data += bottom[0]->offset(1, 0);
  	   }
         top_data += top[0]->offset(1, 0);
  	}
  }

}

template <typename Dtype>
void SubStackFixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(SubStackFixLayer);
REGISTER_LAYER_CLASS(SubStackFix);

}  // namespace caffe
