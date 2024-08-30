#include "../include/frame.h"
#include "../include/pnp_solver.h"
#include "quantized_pair0.h"

#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MATCH_THRESHOLD 0.9
#define MAX_NUM_MATCH 150
#define N 100

// Normalize and compute the L2 norm between the 2 Qint arrays
// we want to compute dot product of q1 and q2 and divide by the norm of q1 and q2
float squared_dist(const int8_t* desc1, const int8_t* desc2) {
    int sum = 0, norm1_squared = 0, norm2_squared = 0;
    for(int i = 0; i < 256; i++) {
        sum += desc1[i] * desc2[i];
        norm1_squared += desc1[i] * desc1[i];
        norm2_squared += desc2[i] * desc2[i];
    }

    return sum / (sqrt(norm1_squared) * sqrt(norm2_squared));
}

void patch_to_grid(int patch, int* grid_x, int* grid_y, int rows, int cols) {
    *grid_x = patch / rows;
    *grid_y = patch % rows;
}

void grid_to_patch(int grid_x, int grid_y, int* patch, int rows, int cols) {
    *patch = grid_x * rows + grid_y;
}

int main() {
    srand(0);

    Frame frame0;
    frame_create(image0_rows, image0_cols, image0_channels, 0,
                 image0_feature_rows, image0_feature_cols, 
                 image0_semi_scale, image0_semi,
                 image0_desc_scale, image0_desc,
                 &frame0);
    Frame frame1;
    frame_create(image1_rows, image1_cols, image1_channels, 0,
                 image1_feature_rows, image1_feature_cols, 
                 image1_semi_scale, image1_semi,
                 image1_desc_scale, image1_desc,
                 &frame1);

    // First compute softmax for frame0
    int num_valid_patches0 = 0;
    int max_indices0[2400] = {0};
    float probs0[2400] = {0};
    compute_softmax(frame0.semi_scale, frame0.semi,
                     &num_valid_patches0,
                     max_indices0, probs0);

    // Then select top 100 features from frame1
    int num_valid_patches1;
    int patches1[N] = {0};
    int indices1[N] = {0};
    float probs1[N] = {0};

    compute_top_N(image1_semi_scale, image1_semi, N, 
                  &num_valid_patches1, patches1, indices1, probs1);

    // ==================== ABOVE THIS POINT IS SETUP ================

    // Define some arbitrary shift and radius
    int shift_x_grid = 4;
    int shift_y_grid = 4;
    int radius_grid = 4;

    float points1[MAX_NUM_MATCH][2];
    float points2[MAX_NUM_MATCH][2];

    // Match features
    int num_matches = 0;
    for(int i = 0; i < num_valid_patches1; i++) {
        int patch1 = patches1[i];
        int x1_grid, y1_grid;
        patch_to_grid(patch1, &x1_grid, &y1_grid, frame1.feature_rows, frame1.feature_cols);

        const int8_t* desc1 = ((const int8_t (*)[256]) frame1.desc)[patch1];

        bool found_match = false;
        int best_index = -1;
        float best_norm = 0;
        int best_x0_grid = 0;
        int best_y0_grid = 0;

        int min_x0_grid = MAX(x1_grid + shift_x_grid - radius_grid, 0);
        int max_x0_grid = MIN(x1_grid + shift_x_grid + radius_grid, frame1.feature_cols - 1);
        int min_y0_grid = MAX(y1_grid + shift_y_grid - radius_grid, 0);
        int max_y0_grid = MIN(y1_grid + shift_y_grid + radius_grid, frame1.feature_rows - 1);

        // printf("%d %d %d %d")

        for(int x0_grid = min_x0_grid; x0_grid <= max_x0_grid; x0_grid++) {
            for(int y0_grid = min_y0_grid; y0_grid <= max_y0_grid; y0_grid++) {
                int patch0;
                grid_to_patch(x0_grid, y0_grid, &patch0, frame0.feature_rows, frame0.feature_cols);

                int index0 = max_indices0[patch0];

                if(index0 == 64) { continue; }

                float prob0 = probs0[patch0];
                // printf("prob0 = %d\n", prob0);
                if(prob0 < 0.2) {
                  continue;
                }

                const int8_t* desc0 = ((const int8_t (*)[256]) frame0.desc)[patch0];
                float dist = squared_dist(desc0, desc1);

                if(dist > MATCH_THRESHOLD * MATCH_THRESHOLD) {
                    if(!found_match || dist > best_norm) {
                      found_match = true;
                      best_index = index0;
                      best_norm = dist;
                      best_x0_grid = x0_grid;
                      best_y0_grid = y0_grid;
                    }
                }
            }
        }

        if(found_match && num_matches < MAX_NUM_MATCH) {
            int index1 = indices1[i];
            int patch1_x = index1 % 8;
            int patch1_y = index1 / 8;

            int x1 = x1_grid * 8 + patch1_x;
            int y1 = y1_grid * 8 + patch1_y;

            int patch0_x = best_index % 8;
            int patch0_y = best_index / 8;

            int x0 = best_x0_grid * 8 + patch0_x;
            int y0 = best_y0_grid * 8 + patch0_y;
            

            points1[num_matches][0] = x0;
            points1[num_matches][1] = y0;
            points2[num_matches][0] = x1;
            points2[num_matches][1] = y1;

            num_matches++;
        }

        if(num_matches >= MAX_NUM_MATCH) {
          break;
        }

    }

    printf("Number of matches: %d\n", num_matches);
    fflush(stdout);

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


    // // Iterate over the search window
    // for(int x1_grid = min_x1_grid; x1_grid <= max_x1_grid; x1_grid++) {
    //   for(int y1_grid = min_y1_grid; y1_grid <= max_y1_grid; y1_grid++) {
    //       int index1 = x1_grid * frame1.feature_rows + y1_grid;

    //       const float* descriptor1 = frame1.descriptors[index1];

    //       const 

    //       if(index1 == -1) {
    //         continue;
    //       }

    //       float prob = image1_feature_scores[index1];
    //       if(prob < 0.4) {
    //         continue;
    //       }

    //       const float* descriptor1 = frame1.descriptors[index1];

    //       float dist = descriptor_distance(descriptor0, descriptor1);

    //       if(dist > MATCH_THRESHOLD) {
    //         if(!found_match || dist < best_norm) {
    //           found_match = true;
    //           best_index = index1;
    //           best_norm = dist;
    //         }
    //       }
    //   }
    // }

  // // Match features
  // int num_matches = 0;
  // for(int i = 0; i < frame0.num_features; i++) {

  //   int x0 = frame0.feature_xs[i];
  //   int y0 = frame0.feature_ys[i];
  //   const float* descriptor0 = frame0.descriptors[i];

  //   int x0_grid = x0 / CELL_SIZE;
  //   int y0_grid = y0 / CELL_SIZE;

  //   bool found_match = false;
  //   int best_index = -1;
  //   float best_norm = 0;

  //   float prob0 = image0_feature_scores[i];
  //   if(prob0 < 0.4) {
  //     continue;
  //   }

  //   int min_x1_grid = MAX(x0_grid + shift_x_grid - radius_grid, 0);
  //   int max_x1_grid = MIN(x0_grid + shift_x_grid + radius_grid, frame1.feature_cols - 1);
  //   int min_y1_grid = MAX(y0_grid + shift_y_grid - radius_grid, 0);
  //   int max_y1_grid = MIN(y0_grid + shift_y_grid + radius_grid, frame1.feature_rows - 1);

  //   // Iterate over the search window
  //   for(int x1_grid = min_x1_grid; x1_grid <= max_x1_grid; x1_grid++) {
  //     for(int y1_grid = min_y1_grid; y1_grid <= max_y1_grid; y1_grid++) {
  //         int index1 = frame1.coords_to_index[y1_grid * frame1.feature_cols + x1_grid];
  //         if(index1 == -1) {
  //           continue;
  //         }

  //         float prob = image1_feature_scores[index1];
  //         if(prob < 0.4) {
  //           continue;
  //         }

  //         const float* descriptor1 = frame1.descriptors[index1];

  //         float dist = descriptor_distance(descriptor0, descriptor1);

  //         if(dist > MATCH_THRESHOLD) {
  //           if(!found_match || dist < best_norm) {
  //             found_match = true;
  //             best_index = index1;
  //             best_norm = dist;
  //           }
  //         }
  //     }
  //   }

  //   int best_x = 0;
  //   int best_y = 0;
  //   if(found_match || num_matches < MAX_NUM_MATCH) {
  //     best_x = frame1.feature_xs[best_index];
  //     best_y = frame1.feature_ys[best_index];

  //     points1[num_matches][0] = x0;
  //     points1[num_matches][1] = y0;
  //     points2[num_matches][0] = best_x;
  //     points2[num_matches][1] = best_y;

  //     num_matches++;
  //   }

  //   if(num_matches >= MAX_NUM_MATCH) {
  //     break;
  //   }
  //   
  // }

  // printf("Number of matches: %d\n", num_matches);

  // // Variables to store the results
  // float best_E[3][3];
  // int best_inliers[10];
  // int num_inliers;

  // // Camera intrinsics
  // float K[3][3] = {{517.306408, 0.0, 318.643040},
  //                  {0.0, 516.469215, 255.313989},
  //                  {0.0, 0.0, 1.0}};
  // 
  // // Run RANSAC to estimate the essential matrix
  // const int num_iterations = 10;
  // const float inlier_threshold = 1.1;
  // ransac_essential_matrix(num_matches, points1, points2, K, 
  //                         num_iterations, inlier_threshold,
  //                         best_E, best_inliers, &num_inliers);
  // 
  // // Recover rotation and translation from the essential matrix
  // float R1[3][3], R2[3][3], t[3];
  // recover_pose_from_essential_matrix(best_E, R1, R2, t);

  // printf("R1: %f %f %f\n", R1[0][0], R1[0][1], R1[0][2]);
  // printf("    %f %f %f\n", R1[1][0], R1[1][1], R1[1][2]);
  // printf("    %f %f %f\n", R1[2][0], R1[2][1], R1[2][2]);

  // printf("R2: %f %f %f\n", R2[0][0], R2[0][1], R2[0][2]);
  // printf("    %f %f %f\n", R2[1][0], R2[1][1], R2[1][2]);
  // printf("    %f %f %f\n", R2[2][0], R2[2][1], R2[2][2]);

  // printf("t: %f %f %f\n", t[0], t[1], t[2]);

  // return 0;
}
