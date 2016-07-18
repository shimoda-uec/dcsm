#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/util/math_functions.hpp"
#include "caffe/layers/unpooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const in, const Dtype* const mask,
    const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const out) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % pooled_width;
    const int h = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int phstart = 
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * height * width;
    const int offsetm = c * height * width;
    const Dtype* const in_slice = in + offset;
    const Dtype* const mask_slice = mask + offsetm;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * width + pw] == h * pooled_width + w) {
            gradient += in_slice[ph * width + pw];
          }
        }
      }
    out[index] = gradient;
  }
}




template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *mask = bottom[1]->gpu_data();
  
  Dtype *top_data = top[0]->mutable_gpu_data();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        top_data);
      break;
  case PoolingParameter_PoolMethod_AVE:
      NOT_IMPLEMENTED;
      break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(UnpoolingLayer);
} //namespace caffe
