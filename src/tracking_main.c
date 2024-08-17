#include "../include/frame.h"
#include "../include/data/tracking/pair0.h"

#include <stdbool.h>
#include <stdio.h>

int main() {
  Frame frame0;
  frame_create(image0_rows, image0_cols, image0_channels, 0,
               image0_num_features, image0_feature_rows, image0_feature_cols, 
               image0_feature_xs, image0_feature_ys,
               image0_feature_descriptors,
               image0_feature_scores, image0_coord_to_index, 
               &frame0);
  Frame frame1;
  frame_create(image1_rows, image1_cols, image1_channels, 0,
               image1_num_features, image1_feature_rows, image1_feature_cols, 
               image1_feature_xs, image1_feature_ys,
               image1_feature_descriptors,
               image1_feature_scores, image1_coord_to_index, 
               &frame1);

  // Define some arbitrary shift and radius
  int shift_x_grid = 4;
  int shift_y_grid = 4;
  int radius_grid = 4;

  // Match features
  int num_matches = 0;
  for(int i = 0; i < frame0.num_features; i++) {

    int x0 = frame0.feature_xs[i];
    int y0 = frame0.feature_ys[i];
    const float* descriptor0 = frame0.descriptors[i];

    int x0_grid = x0 / CELL_SIZE;
    int y0_grid = y0 / CELL_SIZE;

    bool found_match = false;
    int best_index = -1;
    float best_distance = 0;

    // Iterate over the search window
    for(int x1_grid = x0_grid + shift_x_grid - radius_grid; 
            x1_grid <= x0_grid + shift_x_grid + radius_grid; x1_grid++) {
      for(int y1_grid = y0_grid + shift_y_grid - radius_grid; 
              y1_grid <= y0_grid + shift_y_grid + radius_grid; y1_grid++) {
          int index1 = frame1.coords_to_index[y1_grid * frame1.feature_cols + x1_grid];
          if(index1 == -1) {
            continue;
          }
          const float* descriptor1 = frame1.descriptors[index1];

          float distance = descriptor_distance(descriptor0, descriptor1);
          if(!found_match || distance < best_distance) {
            found_match = true;
            best_index = index1;
            best_distance = distance;
          }
      }
    }

    int best_x = 0;
    int best_y = 0;
    if(found_match) {
      best_x = frame1.feature_xs[best_index];
      best_y = frame1.feature_ys[best_index];
      num_matches++;
    }
    
  }

  printf("Number of matches: %d\n", num_matches);

  return 0;
}
