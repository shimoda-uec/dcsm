#include <algorithm>
#include <vector>
#include "caffe/layers/sort_ch_layer.hpp"

namespace caffe {
	
template <typename Dtype>
void SortChLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SortChParameter& sort_ch_param = this->layer_param_.sort_ch_param();
  topk_ = sort_ch_param.topk();
}

template <typename Dtype>
void SortChLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), topk_, 1,1);// 
  //top[1]->Reshape(bottom[0]->num(), topk_, 1,1);// 
}
template <typename Dtype>
void SortChLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();//conv value
  Dtype* sortid = top[0]->mutable_cpu_data();

  //Dtype* sortval = top[1]->mutable_cpu_data();
  std::vector<std::pair<Dtype, int> > bottom_data_vector(bottom[0]->channels());
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int j = 0; j < bottom[0]->channels(); ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[j], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + topk_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < topk_; ++j) {
        sortid[j] = bottom_data_vector[j].second;
        //sortval[j] = bottom_data_vector[j].second;
    }
    sortid += top[0]->offset(1, 0);
    bottom_data += bottom[0]->offset(1, 0);
  }
}

template <typename Dtype>
void SortChLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}



INSTANTIATE_CLASS(SortChLayer);
REGISTER_LAYER_CLASS(SortCh);
}  // namespace caffe
