#include <cfloat>
#include <vector>

#include "caffe/layers/max_normalize_fix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxNormalizeFixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MaxNormalizeParameter mn_param = this->layer_param_.max_normalize_param();
  prior_=mn_param.prior();
}

template <typename Dtype>
void MaxNormalizeFixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}

template <typename Dtype>
void MaxNormalizeFixLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  //const Dtype* eachn = bottom[1]->cpu_data();//conv value
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int inner_num=bottom[0]->height()*bottom[0]->width()*bottom[0]->channels();
  k_.Reshape(1,1,bottom[0]->height(),bottom[0]->width());
  for (int n = 0; n < bottom[0]->num(); ++n) {
	     	mval_=-FLT_MAX;
	     	for(int i=0;i<inner_num;++i){
	     		if(mval_<bottom_data[i]){
	     			mval_=bottom_data[i];
	     		}
	     	}
	     	if(mval_ > 0){
    	     	Dtype k=prior_/mval_;
    	     	caffe_set(inner_num, k, k_.mutable_cpu_data());
    	     	caffe_mul(inner_num, bottom_data, k_.cpu_data(), top_data);
	     	}else{
	     		caffe_set(inner_num, Dtype(0), top_data);
	     	}
            top_data += top[0]->offset(1, 0);
            bottom_data += bottom[0]->offset(1, 0);
  }

}

template <typename Dtype>
void MaxNormalizeFixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}
//#ifdef CPU_ONLY
//STUB_GPU(MaxNormalizeLayer);
//#endif


INSTANTIATE_CLASS(MaxNormalizeFixLayer);
REGISTER_LAYER_CLASS(MaxNormalizeFix);

}  // namespace caffe
