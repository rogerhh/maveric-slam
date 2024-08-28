#pragma once
#include <stdint.h>
#include "descriptor.h"

#define CELL_SIZE 8

typedef struct {
  int rows;     // H
  int cols;     // W
  int channels;
  const char* data;

  // Features
  int num_features;
  int feature_rows;     // H / 8
  int feature_cols;     // W / 8
  const int* feature_xs;
  const int* feature_ys;

  float semi_scale;
  const int8_t* semi;
  float desc_scale;
  const int8_t* desc;

  // const float (*descriptors)[DESCRIPTOR_SIZE];
  // const float* probabilities;

  // const int* coords_to_index;

} Frame;

void frame_create(const int rows, const int cols, const int channels, const char* data, 
                  const int feature_rows, const int feature_cols, 
                  const float semi_scale, const int8_t* semi,
                  const float desc_scale, const int8_t* desc,
                  Frame* frame) {
  frame->rows = rows;
  frame->cols = cols;
  frame->channels = channels;
  frame->data = data;
  frame->feature_rows = feature_rows;
  frame->feature_cols = feature_cols;
  frame->semi_scale = semi_scale;
  frame->semi = semi;
  frame->desc_scale = desc_scale;
  frame->desc = desc;
}
