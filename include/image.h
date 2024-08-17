#pragma once
#include <stdint.h>

typedef struct {
  int rows;     // H
  int cols;     // W
  int channels;
  char* data;

  // Features
  int feature_rows;     // H / 8
  int feature_cols;     // W / 8
  float descriptor_scale;
  uint32_t* descriptors;
  float* probabilities;


} Frame;
