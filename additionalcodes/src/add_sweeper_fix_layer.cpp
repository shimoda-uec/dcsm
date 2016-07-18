#include <algorithm>
#include <vector>
#include "caffe/layers/add_sweeper_fix_layer.hpp"

namespace caffe {
using std::sort;

template <typename Dtype>
void AddSweeperFixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AddSweeperParameter sweeper_param = this->layer_param_.add_sweeper_param();
  sweepern_=sweeper_param.sweepern();
}

template <typename Dtype>
void AddSweeperFixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  top[0]->Reshape(1, channels_, 1,1);// each threshold 
  top[1]->Reshape(1, 1, 1,1);// each threshold num
  top[2]->Reshape(sweepern_, 1, 1,1);// each threshold num  
}
template <typename Dtype>
void AddSweeperFixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* sortid = bottom[0]->cpu_data();//conv value
  Dtype* sweeper = top[0]->mutable_cpu_data();
  Dtype* sweepern = top[1]->mutable_cpu_data();
  Dtype* sweeperid = top[2]->mutable_cpu_data();
  sweeperid_.clear();
  caffe_set(channels_,Dtype(0),sweeper);
  for (int n = 0; n < 1; ++n) {
  	sweepern[n]=sweepern_;
      	for(int i=0;i<sweepern_;++i){
          	sweeperid_.push_back(sortid[i]);
           	const int id=sortid[i];
	    	sweeper[id]=1;
  	}
  	sweeper +=top[0]->offset(1, 0);
       sortid += bottom[0]->offset(1, 0);
  } 
  sort(sweeperid_.begin(), sweeperid_.end());
  int sid=0;
  for(int i=0;i<1;++i){
  	for (int j=0;j<sweepern_;++j){
		sweeperid[j]=sweeperid_[j];
  	}
  }
}

template <typename Dtype>
void AddSweeperFixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(AddSweeperFixLayer);
REGISTER_LAYER_CLASS(AddSweeperFix);
}  // namespace caffe
