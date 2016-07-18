#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/kernel_max_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KernelForward(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height, const int width, Dtype* const out) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    //const int offset = (n * channels + c) * h * w;
    //const Dtype* const in_slice = in + offset;
    Dtype km=-FLT_MAX;
    for (int c=0; c < channels; ++c) {
        if(in[(n*channels+c)*height*width+h*width+w] > km){
           km=in[(n*channels+c)*height*width+h*width+w];
           //kmid=c;
        }
    }
    out[index] = km;
  }
}

template <typename Dtype>
void KernelMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();//derivative
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
    KernelForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void KernelMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(KernelMaxLayer);
}  // namespace caffe
