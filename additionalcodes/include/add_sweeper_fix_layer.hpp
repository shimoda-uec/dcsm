#ifndef CAFFE_ADD_SWEEPER_FIX_LAYER_HPP_
#define CAFFE_ADD_SWEEPER_FIX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class AddSweeperFixLayer : public Layer<Dtype> {
 public:
  explicit AddSweeperFixLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "AddSweeperFix"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //  NOT_IMPLEMENTED;
  //}
  int channels_;
  int objn_;
  int sweepern_;
  vector<int> sweeperid_;
  vector<int> objid_;
  vector<int> overlapid_;

};

}  // namespace caffe

#endif  // CAFFE_THRESHOLD_LAYER_HPP_
