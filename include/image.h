#pragma once

typedef struct {
  int rows;
  int cols;
  int channels;
  char* data;

  // Features
  char* probabilities;

  int feature_rows;     // H / 8
  int feature_cols;     // W / 8

  int num_features;
  int* feature_xs;
  int* feature_ys;
  int* feature_grid_xs;
  int* feature_grid_ys;
  Descriptors** descriptors;

} Frame;
