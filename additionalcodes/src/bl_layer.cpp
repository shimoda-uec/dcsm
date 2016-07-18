#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/bl_layer.hpp"

namespace caffe {
template <typename Dtype>
void BlLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BlParameter bl_param = this->layer_param_.bl_param();
  newsize_=bl_param.newsize();
}

template <typename Dtype>
void BlLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), newsize_,newsize_);
}

template <typename Dtype>
void BlLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    basesize_=bottom[0]->height();
    const Dtype* bottom_data = bottom[0]->cpu_data();
 	Dtype* top_data = top[0]->mutable_cpu_data();
      	double scale2 = (double)newsize_/basesize_;
      	for (int n = 0; n < bottom[0]->num(); ++n) {
          for (int y = 0; y < newsize_; ++y ){
           for (int x = 0; x < newsize_; ++x ){
		     	int x0 = (int)x/scale2;
		     	int y0 = (int)y/scale2;
              	double a = (double)x/scale2 - x0;
		       	double b = (double)y/scale2 - y0;
           		int x1 = x0 + 1 >= basesize_ ? x0 : x0 + 1;
           		int y1 = y0 + 1 >= basesize_ ? y0 : y0 + 1;
           	
		        for(int k =0;k<bottom[0]->channels();++k){
		      		top_data[k*newsize_*newsize_+y*newsize_+x] = (1-a)*(1-b)*bottom_data[k*newsize_*newsize_+y0 * basesize_ + x0] 
		                               + a*(1-b)*bottom_data[k*newsize_*newsize_+y0 * basesize_ + x1]
		                                    + (1-a)*b*bottom_data[k*newsize_*newsize_+y1 * basesize_ + x0]
		                                        + a*b*bottom_data[k*newsize_*newsize_+y1 * basesize_ + x1];
		        }
           }
         }
        bottom_data += bottom[0]->offset(1, 0);
        top_data += top[0]->offset(1, 0);
    }//n
}
template <typename Dtype>
void BlLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(BlLayer);
REGISTER_LAYER_CLASS(Bl);

}  // namespace caffe

