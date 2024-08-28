#include "top_N.h"
#include <float.h>
#include <stdio.h>
#include <stdbool.h>

#define P 5

// Compute exp(scale x) using Taylor's expansion to the pth degree
// We want to compute e^x = 1 + scale x / 1 + scale^2 x^2 / 2 + ...
//                        = 1 + scale / 1 * x + scale^2 / 2 * x^2 + ...
float approx_exp(float* scale_poly, int8_t x) {
    float sum = 1.0f;
    int x_poly = x;
    for(int i = 1; i < P; i++) {
        sum += scale_poly[i] * x_poly;
        x_poly *= x;
    }
    return sum;
}

void approx_softmax(float* scale_poly, int8_t semi_row[65], int* max_index, float* max_prob) {
    // First compute approximate softmax. This is okay to do 
    // because most entries are large negative values
    // We will skip over all negative values

    *max_index = 64;
    float max_exp_x = 0;
    float denominator = FLT_MIN;

    for(int i = 0; i < 65; i++) {
        // printf("%d ", semi_row[i]);
        // Skip negative values
        if(semi_row[i] < 0) {
            continue;
        }
        float exp_x = approx_exp(scale_poly, semi_row[i]);

        if(i != 64 && exp_x > max_exp_x) {
            max_exp_x = exp_x;
            *max_index = i;
        }

        denominator += exp_x;
    }
    // printf("\n");

    *max_prob = max_exp_x / denominator;

}

#define MAX_VALID_FEATURES 1000

void compute_top_N(float scale, int8_t semi[2400][65], int N, 
                   int* num_selected, int* N_patches, int* N_indices, float* N_probs) {
    
    // First compute the sequence {1, scale / 1, scale^2 / 2!, scale^3 / 3!, ...}
    float scale_poly[P] = {0};
    scale_poly[0] = 1;
    for(int i = 1; i < P; i++) {
        scale_poly[i] = scale_poly[i - 1] * scale / i;
    }

    // First go through each row and compute probability of each row and max index
    // Also find the range of the probabilities and the number of valid probabilities
    int max_indices[MAX_VALID_FEATURES] = {0};
    float probs[MAX_VALID_FEATURES] = {0};
    float max_prob = 0, min_prob = FLT_MAX;
    int num_valid = 0;
    int valid_patches[MAX_VALID_FEATURES] = {0};

    for(int patch = 0; patch < 2400; patch++) {
        int max_index;
        float prob;
        approx_softmax(scale_poly, semi[patch], &max_index, &prob);
        if(max_index != 64) {
            valid_patches[num_valid] = patch;
            max_indices[num_valid] = max_index;
            probs[num_valid] = prob;

            if(probs[patch] > max_prob) {
                max_prob = probs[patch];
            }
            if(probs[patch] < min_prob) {
                min_prob = probs[patch];
            }

            num_valid++;
        }
    }

    if(num_valid <= N) {
        *num_selected = num_valid;
        for(int i = 0; i < num_valid; i++) {
            N_patches[i] = valid_patches[i];
            N_indices[i] = max_indices[i];
            N_probs[i] = probs[i];
        }
        return;
    }

    float split = N / (float) num_valid;
    float threshold = max_prob * split + min_prob * (1 - split);

    // printf("Max prob = %f\n", max_prob);
    // printf("Min prob = %f\n", min_prob);
    // printf("Select threshold = %f\n", threshold);

    *num_selected = 0;
    for(int i = 0; i < num_valid; i++) {

        int patch = valid_patches[i];
        float prob = probs[i];

        if(prob >= threshold) {
            N_patches[*num_selected] = patch;
            N_indices[*num_selected] = max_indices[i];
            N_probs[*num_selected] = prob;

            (*num_selected)++;

            if(*num_selected >= N) {
                return;
            }
        }

    }
}

void compute_softmax(float scale, int8_t semi[2400][65],
                     int* num_valid, int* patch_to_valid_index, 
                     int* N_indices, float* N_probs) {
    
//     // First compute the sequence {1, scale / 1, scale^2 / 2!, scale^3 / 3!, ...}
//     float scale_poly[P] = {0};
//     scale_poly[0] = 1;
//     for(int i = 1; i < P; i++) {
//         scale_poly[i] = scale_poly[i - 1] * scale / i;
//     }
// 
//     // First go through each row and compute probability of each row and max index
//     // Also find the range of the probabilities and the number of valid probabilities
//     int patch_to_valid_index[2400] = {0};
//     int max_indices[MAX_VALID_FEATURES] = {0};
//     float probs[MAX_VALID_FEATURES] = {0};
//     float max_prob = 0, min_prob = FLT_MAX;
//     int num_valid = 0;
//     int valid_patches[MAX_VALID_FEATURES] = {0};
// 
//     for(int patch = 0; patch < 2400; patch++) {
//         int max_index;
//         float prob;
//         approx_softmax(scale_poly, semi[patch], &max_index, &prob);
//         if(max_index != 64) {
//             patch_to_valid_index[patch] = num_valid;
// 
//             valid_patches[num_valid] = patch;
//             max_indices[num_valid] = max_index;
//             probs[num_valid] = prob;
// 
//             if(probs[patch] > max_prob) {
//                 max_prob = probs[patch];
//             }
//             if(probs[patch] < min_prob) {
//                 min_prob = probs[patch];
//             }
// 
//             num_valid++;
//         }
//     }
}
