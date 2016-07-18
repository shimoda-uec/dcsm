#include <algorithm>
#include <vector>

#include "caffe/layers/guided_relu_set_layer.hpp"


namespace caffe {

template <typename Dtype>
__global__ void GuidedReLUSetForward(const int n, const Dtype* in,const Dtype* in2,
    Dtype* out, int i) {
  CUDA_KERNEL_LOOP(index, n) {
       for(int e=0;e<i;++e){
         out[e*n+index] = in[e*n+index]*(in2[0*n+index] > 0)*(in[e*n+index]>0);
        }
  }
}
template <typename Dtype>
void GuidedReLUSetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();//derivative
  const Dtype* bottom_data2 = bottom[1]->gpu_data();//conv value
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int inner_num=bottom[1]->channels()*bottom[1]->height()*bottom[1]->width();
  GuidedReLUSetForward<Dtype><<<CAFFE_GET_BLOCKS(inner_num), CAFFE_CUDA_NUM_THREADS>>>(
               inner_num, bottom_data,bottom_data2,top_data,bottom[0]->num());
            CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void GuidedReLUSetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(GuidedReLUSetLayer);
}  // namespace caffe
