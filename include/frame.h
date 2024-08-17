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
  float* descriptors;
  float* probabilities;

} Frame;

void frame_create(int rows, int cols, int channels, char* data, 
                  int feature_rows, int feature_cols, 
                  float* descriptors, float* probabilities,
                  Frame* frame) {
    frame->rows = rows;
    frame->cols = cols;
    frame->channels = channels;
    frame->data = data;
    frame->feature_rows = feature_rows;
    frame->feature_cols = feature_cols;
    frame->descriptors = descriptors;
    frame->probabilities = probabilities;
}
