#include <cfloat>
#include <vector>

#include "caffe/layers/sub_fix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SubFixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SubFixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num()*bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}
template <typename Dtype>
void SubFixLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int inner_num=bottom[0]->height()*bottom[0]->width()*bottom[0]->channels();
  for (int n = 0; n < 1; ++n) {
  	for (int e1=0;e1<bottom[0]->num();++e1){
  	  for (int e2=0;e2<bottom[0]->num();++e2){
	     if(e1!=e2){
	  	    caffe_copy(inner_num, bottom_data + bottom[0]->offset(e1, 0), top_data);
	  	    caffe_axpy(inner_num, Dtype(-1.0), bottom_data + bottom[0]->offset(e2, 0) , top_data);
            top_data += top[0]->offset(1, 0);
	      }
  	  	  else{
	     	caffe_set(inner_num, Dtype(0), top_data);
            top_data += top[0]->offset(1, 0);
  	  	  }
  	  }
  	}
  	bottom_data += bottom[0]->offset(bottom[0]->num(), 0);
  }

}

template <typename Dtype>
void SubFixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}
//#ifdef CPU_ONLY
//#STUB_GPU(SubLayer);
//#endif


INSTANTIATE_CLASS(SubFixLayer);
REGISTER_LAYER_CLASS(SubFix);

}  // namespace caffe
