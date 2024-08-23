#include "../include/frame.h"
#include "../include/pnp_solver.h"
#include "../include/data/tracking/pair0.h"

#include <stdbool.h>
#include <stdio.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MATCH_THRESHOLD 0.9
#define MAX_NUM_MATCH 2000

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

  float points1[MAX_NUM_MATCH][2];
  float points2[MAX_NUM_MATCH][2];

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
    float best_norm = 0;

    float prob0 = image0_feature_scores[i];
    if(prob0 < 0.4) {
      continue;
    }

    int min_x1_grid = MAX(x0_grid + shift_x_grid - radius_grid, 0);
    int max_x1_grid = MIN(x0_grid + shift_x_grid + radius_grid, frame1.feature_cols - 1);
    int min_y1_grid = MAX(y0_grid + shift_y_grid - radius_grid, 0);
    int max_y1_grid = MIN(y0_grid + shift_y_grid + radius_grid, frame1.feature_rows - 1);

    // Iterate over the search window
    for(int x1_grid = min_x1_grid; x1_grid <= max_x1_grid; x1_grid++) {
      for(int y1_grid = min_y1_grid; y1_grid <= max_y1_grid; y1_grid++) {
          int index1 = frame1.coords_to_index[y1_grid * frame1.feature_cols + x1_grid];
          if(index1 == -1) {
            continue;
          }

          float prob = image1_feature_scores[index1];
          if(prob < 0.4) {
            continue;
          }

          const float* descriptor1 = frame1.descriptors[index1];

          float dist = descriptor_distance(descriptor0, descriptor1);

          if(dist > MATCH_THRESHOLD) {
            if(!found_match || dist < best_norm) {
              found_match = true;
              best_index = index1;
              best_norm = dist;
            }
          }
      }
    }

    int best_x = 0;
    int best_y = 0;
    if(found_match || num_matches < MAX_NUM_MATCH) {
      best_x = frame1.feature_xs[best_index];
      best_y = frame1.feature_ys[best_index];

      points1[num_matches][0] = x0;
      points1[num_matches][1] = y0;
      points2[num_matches][0] = best_x;
      points2[num_matches][1] = best_y;

      num_matches++;
    }

    if(num_matches >= MAX_NUM_MATCH) {
      break;
    }
    
  }

  printf("Number of matches: %d\n", num_matches);

  // Variables to store the results
  float best_E[3][3];
  int best_inliers[10];
  int num_inliers;

  // Camera intrinsics
  float K[3][3] = {{517.306408, 0.0, 318.643040},
                   {0.0, 516.469215, 255.313989},
                   {0.0, 0.0, 1.0}};
  
  // Run RANSAC to estimate the essential matrix
  const int num_iterations = 10;
  const float inlier_threshold = 1.1;
  ransac_essential_matrix(num_matches, points1, points2, K, 
                          num_iterations, inlier_threshold,
                          best_E, best_inliers, &num_inliers);
  
  // Recover rotation and translation from the essential matrix
  float R1[3][3], R2[3][3], t[3];
  recover_pose_from_essential_matrix(best_E, R1, R2, t);

  printf("R1: %f %f %f\n", R1[0][0], R1[0][1], R1[0][2]);
  printf("    %f %f %f\n", R1[1][0], R1[1][1], R1[1][2]);
  printf("    %f %f %f\n", R1[2][0], R1[2][1], R1[2][2]);

  printf("R2: %f %f %f\n", R2[0][0], R2[0][1], R2[0][2]);
  printf("    %f %f %f\n", R2[1][0], R2[1][1], R2[1][2]);
  printf("    %f %f %f\n", R2[2][0], R2[2][1], R2[2][2]);

  printf("t: %f %f %f\n", t[0], t[1], t[2]);

  return 0;
}
