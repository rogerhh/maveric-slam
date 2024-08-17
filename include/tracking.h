#pragma once

void track(const Frame* last_frame, const Frame* current_frame, const int x_shift, const int y_shift, const int window_size, const float threshold, Transform* transform) {
  if(last_frame == nullptr) {
    transform = Identity();
    return;
  }

  int num_matched_features;
  int src_xs[];
  int src_ys[];
  int dst_xs[];
  int dst_ys[];

  for(int i = 0; i < last_frame->num_features; i++) {
    int feature_grid_x = last_frame->feature_grid_xs[i];
    int feature_grid_y = last_frame->feature_grid_ys[i];
    Descriptor* prev_descriptor = last_frame->descriptors[feature_grid_y + feature_grid_x * feature_rows];

    int half_window_size = (window_size - 1) / 2;
    int min_grid_x = MAX(0, feature_grid_x + x_shift - half_window_size);
    int max_grid_x = MIN(feature_cols, feature_grid_x + half_window_size);
    int min_grid_y = MAX(0, feature_grid_y + y_shift - half_window_size);
    int max_grid_y = MIN(feature_rows, feature_grid_y + half_window_size);

    float best_score = 0.0;

    for(int grid_r = min_grid_y; grid_r < max_grid_y; grid_r++) {
      for(int grid_c = min_grid_x; grid_c < max_grid_x; grid_c++) {
        Descriptor* descriptor = current_frame->descriptors[grid_r + feature_grid_c * feature_rows];

        float score = compare_descriptors(prev_descriptor, descriptor);

        if(score > best_score) {
          best_score = score;
        }
      }
    }

    if(best_score > threshold) {
      num_matched_features++;
    }
  }

  float jacobian = {0};
  for(int i = 0; i < num_matched_features; i++) {
    // populate the jacobian
  } 

  // Compute J^TJ
  // Solve linear system
  //
  
}
