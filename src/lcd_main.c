#include "vocabulary.h"
#include "quantized_pair0.h"
#include "top_N.h"
#include "gemmini_functions_cpu.h"

#include <stdio.h>
#include <stdbool.h>

#define N 100

const int8_t count_lookup[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, };

void get_binary_descriptor(float scale, int* feature, int* binary_descriptor, int size)
{
    if(scale > 0) {
        for(int i = 0; i < size; i++) {
            binary_descriptor[i] = 0;
            for(int j = 0; j < 32; j++) {
                int index = i * 32 + j;
                binary_descriptor[i] <<= 1;
                if(feature[index] > 0) {
                    binary_descriptor[i] += 1;
                }
            }
        }
    }
    else {
        for(int i = 0; i < size; i++) {
            binary_descriptor[i] = 0;
            for(int j = 0; j < 32; j++) {
                int index = i * 32 + j;
                binary_descriptor[i] <<= 1;
                if(feature[index] <= 0) {
                    binary_descriptor[i] += 1;
                }
            }
        }
    }
}

int count_matching_bits(int* binary_desc1, int* binary_desc2, int size) {
    int count = 0;
    for(int i = 0; i < size; i++) {
        int matching = ~(binary_desc1[i] ^ binary_desc2[i]);
        for(int j = 0; j < 4; j++) {
            int partial_match = (matching >> (j * 8)) & 255;
            count += count_lookup[partial_match];
        }
    }
    return count;
}


int main() {

    // First select top 500 features
    int patches[N] = {0};
    int indices[N] = {0};
    float probs[N] = {0};
    int num_valid_patches;

    compute_top_N(image0_semi_scale, image0_semi, N, 
                  &num_valid_patches, patches, indices, probs);

    printf("Num valid patches: %d\n", num_valid_patches);
    for(int i = 0; i < num_valid_patches; i++) {
        printf("Patch index: %d, index: %d, probability: %f\n", patches[i], indices[i], probs[i]);
    }

    printf("before construct features\n");
    fflush(stdout);

    // Construct the features matrix
    float feature_scale = image0_desc_scale;
    int8_t features[N][256] = {0};

    for(int i = 0; i < num_valid_patches; i++) {
        int patch = patches[i];
        for(int j = 0; j < 256; j++) {
            features[i][j] = image0_desc[patch][j];
        }
    }

    int8_t scores[N][num_base_nodes];

    // We have an Nx256 matrix and an 10x256 matrix
    matmul(N, num_base_nodes, 256,                  // I, J, K
           features, base_descriptors, scores,      // A, B, C
           256, 256, 10,                            // strideA, strideB, strideC
           feature_scale, 1.0,                      // scaleA, scaleB
           false, true);                            // transposeA, transposeB
                                                    // mvout scale should be 1/256
    int sel_base_nodes[N] = {0};
    for(int i = 0; i < num_valid_patches; i++) {
        float max_score = 0;
        for(int j = 0; j < num_base_nodes; j++) {
            float score = scores[i][j];
            score = scale_arr[j] * score + 256 * bias_arr[j];

            if(score > max_score) {
                max_score = score;
                sel_base_nodes[i] = j;
            }
        }
    }

    // Convert selected features into binary descriptors
    int binary_features[N][8] = {0};
    for(int i = 0; i < num_valid_patches; i++) {
        get_binary_descriptor(image0_desc_scale, features[i], binary_features[i], 8);
    }

    
    // Do flattened tree traversal
    for(int i = 0; i < num_valid_patches; i++) {
        int base_node = sel_base_nodes[i];
        int best_match = 0;
        int best_wid = 0;
        for(int wid = 0; wid < words_per_base_node; wid++) {
            int match = count_matching_bits(binary_features[i], leaf_descriptors[base_node][wid], 8);
            if(match > best_match) {
                best_match = match;
                best_wid = wid;
            }
        }

        printf("Patch: %d, word: %d\n", i, best_wid);

    }
}
