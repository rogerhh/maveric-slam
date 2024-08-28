#pragma once

#include <stdint.h>

// Given a (W, H, C=65) int array and a scale
// 1. compute the approximate softmax across the channel dimension
// 2. set indices to be approximately the top N probabilities
void compute_top_N(float scale, int8_t semi[2400][65], int N, 
                   int* num_selected, int* N_patches, int* N_indices, float* N_probs);
