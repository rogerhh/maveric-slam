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
#define N 500

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

void patch_index_to_patch_coords(int index, int* patch_x, int* patch_y) {
    *patch_x = index % 8;
    *patch_y = index / 8;
}

int main() {
    srand(0);

    Frame frame1;
    frame_create(image1_rows, image1_cols, image1_channels, 0,
                 image1_feature_rows, image1_feature_cols, 
                 image1_semi_scale, image1_semi,
                 image1_desc_scale, image1_desc,
                 &frame1);

    // First compute softmax for frame0
    int num_valid_patches1 = 0;
    int max_indices1[2400] = {0};
    float probs1[2400] = {0};
    bool processed[2400] = {0};
    compute_softmax(frame1.semi_scale, frame1.semi,
                     &num_valid_patches1,
                     max_indices1, probs1);

    const int x_grid_max = 80;
    const int y_grid_max = 30;

    for(int x_grid_i = 0; x_grid_i <= x_grid_max; x_grid_i++) {
        for(int y_grid_i = 0; y_grid_i <= y_grid_max; y_grid_i++) {

            // First gather the valid patches in the 4 quandrants
            int num_valid_patches = 0;
            int patches[4] = {0};
            float probs[4] = {0};
            int xs[4] = {0};
            int ys[4] = {0};

            for(int x_delta = -1; x_delta <= 0; x_delta++) {
                int x_grid = x_grid_i + x_delta;
                if(x_grid < 0 || x_grid >= x_grid_max) { continue; }

                for(int y_delta = -1; y_delta <= 0; y_delta++) {
                    int y_grid = y_grid_i + y_delta;
                    if(y_grid < 0 || y_grid >= y_grid_max) { continue; }

                    int patch; 
                    grid_to_patch(x_grid, y_grid, &patch, frame1.feature_rows, frame1.feature_cols);

                    int index = max_indices1[patch];
                    if(index == 64) { continue; }

                    int patch_x, patch_y;
                    patch_index_to_patch_coords(index, &patch_x, &patch_y);

                    if(x_delta == -1 && patch_x < 2) { continue; } 
                    if(x_delta == 0 && patch_x >= 6) { continue; } 
                    if(y_delta == -1 && patch_y < 2) { continue; } 
                    if(y_delta == 0 && patch_y >= 6) { continue; } 
                    
                    patches[num_valid_patches] = patch;
                    probs[num_valid_patches] = probs1[patch];

                    xs[num_valid_patches] = x_grid * 8 + patch_x;
                    ys[num_valid_patches] = y_grid * 8 + patch_y;

                    num_valid_patches++;
                }
            }

            // Find the max probility in the 4 quandrants and do NMS
            while(1) {
                float max_prob = 0;
                int max_index = -1;

                for(int i = 0; i < num_valid_patches; i++) {
                    if(patches[i] > 0 && probs[i] > max_prob) {
                        max_prob = probs[i];
                        max_index = i;
                    }
                }

                if(max_index == -1) { break; }


                for(int i = 0; i < num_valid_patches; i++) {
                    if(patches[i] >= 0 && probs[i] > max_prob) {
                        max_prob = probs[i];
                        max_index = i;
                    }
                }

                for(int i = 0; i < num_valid_patches; i++) {
                    if(i == max_index) { continue; }
                    if(patches[i] < 0) { continue; }

                    int x_diff = abs(xs[max_index] - xs[i]);
                    int y_diff = abs(ys[max_index] - ys[i]);

                    if(x_diff < 4 && y_diff < 4) {
                        int suppress_patch = patches[i];
                        max_indices1[suppress_patch] = 64;
                        probs1[suppress_patch] = 64;

                        printf("(%d %d) suppressing (%d %d)\n", xs[max_index], ys[max_index], xs[i], ys[i]);

                        patches[i] = -1;
                        probs[i] = -1;
                    }
                }

                probs[max_index] = -1;
                patches[max_index] = -1;

            }



        }
    }

    for(int patch = 0; patch < 2400; patch++) {
        int index = max_indices1[patch];
        if(index == 64) { continue; }

        int x_grid, y_grid;
        patch_to_grid(patch, &x_grid, &y_grid, frame1.feature_rows, frame1.feature_cols);


        int patch_x, patch_y;
        patch_index_to_patch_coords(index, &patch_x, &patch_y);

        int x = x_grid * 8 + patch_x;
        int y = y_grid * 8 + patch_y;

        printf("%d %d\n", x, y);
    }

}
