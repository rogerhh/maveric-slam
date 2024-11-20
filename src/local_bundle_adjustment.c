/**
 * local_bundle_adjustment.c
 * Given the Jacobian matrix J and residual vector f, solve the normal equations 
 * (J^TJ + lambda*I)x = J^Tf
 * Each block row in J represents a factor, each factor connects a ldmk and a pose
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "gemmini_functions_cpu.h"

#define NUM_LDMKS 1000
#define NUM_POSES 8
#define LDMK_CHUNK 4
#define POSE_DIM 6
#define LDMK_DIM 3
#define FACTOR_WIDTH (POSE_DIM + LDMK_DIM + 1)
#define FACTOR_HEIGHT 2
#define TOTAL_POSE_DIM (NUM_POSES * POSE_DIM)
#define SUBDIAG_HEIGHT (TOTAL_POSE_DIM + 1)
#define TOTAL_LDMK_DIM (LDMK_CHUNK * LDMK_DIM)

#define min(a, b) ((a) < (b) ? (a) : (b))

void print_col_major_matrix(float* matrix, int rows, int cols, int stride) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[j * stride + i]);
        }
        printf("\n");
    }
}

// Compute C = alpha * A + beta * B
// A, B, C are all col-major matrices
void matrix_add(float* A, float* B, float* C,  
                int rows, int cols,
                int strideA, int strideB, int strideC,
                float alpha, float beta) {
    for(int j = 0; j < cols; j++) {
        for(int i = 0; i < rows; i++) {
            C[j * strideC + i] = alpha * A[j * strideA + i] + beta * B[j * strideB + i];
        }
    }
}

void invert_3x3(float* matrix, int stride) {
    float mat[9];
    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            mat[j * 3 + i] = matrix[j * stride + i];
        }
    }
    float inv_mat[9];
    float det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) - 
                mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
                mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    inv_mat[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / det;
    inv_mat[1] = (mat[2] * mat[7] - mat[1] * mat[8]) / det;
    inv_mat[2] = (mat[1] * mat[5] - mat[2] * mat[4]) / det;
    inv_mat[3] = (mat[5] * mat[6] - mat[3] * mat[8]) / det;
    inv_mat[4] = (mat[0] * mat[8] - mat[2] * mat[6]) / det;
    inv_mat[5] = (mat[2] * mat[3] - mat[0] * mat[5]) / det;
    inv_mat[6] = (mat[3] * mat[7] - mat[4] * mat[6]) / det;
    inv_mat[7] = (mat[1] * mat[6] - mat[0] * mat[7]) / det;
    inv_mat[8] = (mat[0] * mat[4] - mat[1] * mat[3]) / det;

    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            matrix[j * stride + i] = inv_mat[j * 3 + i];
        }
    }
}

void invert_block_diagonal_matrix(float* matrix, int dim, int block_size) {
    assert(block_size == 3);
    assert(dim % block_size == 0);
    for(int I = 0; I < dim; I += block_size) {
        invert_3x3(matrix + I * dim + I, dim);
    }
}

void cholesky(float* matrix, int dim, int stride) {
    printf("Not implemented due to laziness XD\n");
}

void initialize_random_matrix(float* matrix, int rows, int cols) {
    for(int j = 0; j < cols; j++) {
        for(int i = 0; i < rows; i++) {
            matrix[j * rows + i] = i * 10 + j;
        }
    }
}

void zero_matrix(float* matrix, int rows, int cols) {
    // This loop can be replaced by memset
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = 0;
    }
}

void initialize_SPD_matrix(float* matrix, int dim) {
    float A[dim * dim];
    zero_matrix(matrix, dim, dim);
    initialize_random_matrix(A, dim, dim);

    matmul2(dim, dim, dim,                  // I, J, K
            A, A, matrix, matrix,           // A, B, D, C
            dim, dim, dim, dim,             // strideA, strideB, strideD, strideC
            1, 1, 0,                        // scaleA, scaleB, scaleD
            true, false);                   // transposeA, transposeB
}

void zero_block_diagonal_matrix(float* matrix, int dim, int block_size) {
    assert(dim % block_size == 0);
    for(int I = 0; I < dim; I += block_size) {
        for(int i = 0; i < block_size; i++) {
            for(int j = 0; j < block_size; j++) {
                matrix[(I + i) * dim + I + j] = 0;
            }
        }
    }
}

int main() {
    srand(0);

    // Initialize the dense pose matrix
    float C[SUBDIAG_HEIGHT * SUBDIAG_HEIGHT] = {0};

    // Allocate the diagonal chunk of the Hessian matriz
    // This is a dense TOTAL_LDMK_DIM x TOTAL_LDMK_DIM matrix
    // But only the block diagonal elements are non-zero
    float A_chunk[TOTAL_LDMK_DIM * TOTAL_LDMK_DIM] = {0};

    // Allocate the dense LP matrix
    // This is a dense SUBDIAG_HEIGHT x TOTAL_LDMK_DIM matrix
    float B_chunk[SUBDIAG_HEIGHT * TOTAL_LDMK_DIM] = {0};
    float B_Ainv_chunk[SUBDIAG_HEIGHT * TOTAL_LDMK_DIM] = {0};


    // Allocate factor matrix that can be reused
    // J_factor_chunk contains all the factors for a single chunk
    float J_factor_chunk[NUM_POSES * LDMK_CHUNK][FACTOR_HEIGHT * FACTOR_WIDTH];
    float H_factor[FACTOR_WIDTH * FACTOR_WIDTH];

    for(int chunk_start = 0; chunk_start < NUM_LDMKS; chunk_start += LDMK_CHUNK) {
        int chunk_end = min(chunk_start + LDMK_CHUNK, NUM_LDMKS);

        /*** DONT MEASURE THIS START ***/
        zero_block_diagonal_matrix(A_chunk, TOTAL_LDMK_DIM, LDMK_DIM);
        zero_matrix(B_chunk, SUBDIAG_HEIGHT, TOTAL_LDMK_DIM);
        initialize_random_matrix(J_factor_chunk, 
                                 FACTOR_HEIGHT * NUM_POSES * LDMK_CHUNK, 
                                 FACTOR_WIDTH);
        /*** DONT MEASURE THIS END ***/

        for(int chunk_i = 0; chunk_i < LDMK_CHUNK; chunk_i++) {
            int ldmk_id = chunk_start + chunk_i;
            for(int pose_id = 0; pose_id < NUM_POSES; pose_id++) {
                int pose_idx = pose_id * POSE_DIM;
                int ldmk_idx = chunk_i * LDMK_DIM;

                float* J_factor = J_factor_chunk[pose_id * chunk_i];

                // Compute H_factor = J_factor^T * J_factor
                // This one does not have to be mapped onto accelerator
                matmul2(FACTOR_WIDTH, FACTOR_WIDTH, FACTOR_HEIGHT, // I, J, K
                        J_factor, J_factor, H_factor, H_factor,    // A, B, D, C
                        FACTOR_HEIGHT, FACTOR_HEIGHT,              // strideA, strideB
                        FACTOR_WIDTH, FACTOR_WIDTH,                // strideD, strideC
                        1, 1, 0,                                   // scaleA, scaleB, scaleD
                        false, true);                              // transposeA, transposeB
                                                                   // J_factor is column major

                // Scatter results to the Hessian matrix
                // Add H_LL
                matrix_add(H_factor,                                  // A
                           A_chunk + ldmk_idx * (TOTAL_LDMK_DIM + 1), // B
                           A_chunk + ldmk_idx * (TOTAL_LDMK_DIM + 1), // C
                           LDMK_DIM, LDMK_DIM,                        // rows, cols   
                           FACTOR_WIDTH,                              // strideA
                           TOTAL_LDMK_DIM,                            // strideB
                           TOTAL_LDMK_DIM,                            // strideC  
                           1, 1);                                     // alpha, beta
                // Add H_PL
                matrix_add(H_factor + LDMK_DIM,                            // A  
                           B_chunk + pose_idx + ldmk_idx * SUBDIAG_HEIGHT, // B
                           B_chunk + pose_idx + ldmk_idx * SUBDIAG_HEIGHT, // B
                           POSE_DIM, LDMK_DIM,                             // rows, cols
                           FACTOR_WIDTH,                                   // strideA
                           SUBDIAG_HEIGHT,                                 // strideB
                           SUBDIAG_HEIGHT,                                 // strideC
                           1, 1);                                          // alpha, beta
                // Add H_Pf
                matrix_add(H_factor + FACTOR_WIDTH - 1,                   // A
                           B_chunk + (ldmk_idx + 1) * SUBDIAG_HEIGHT - 1, // B
                           B_chunk + (ldmk_idx + 1) * SUBDIAG_HEIGHT - 1, // C
                           1, LDMK_DIM,                                   // rows, cols
                           FACTOR_WIDTH,                                  // strideA
                           SUBDIAG_HEIGHT,                                // strideB
                           SUBDIAG_HEIGHT,                                // strideC
                           1, 1);                                         // alpha, beta
                // Add H_PP
                matrix_add(H_factor + LDMK_DIM * (FACTOR_WIDTH + 1), // A
                           C + pose_idx * (SUBDIAG_HEIGHT + 1),      // B
                           C + pose_idx * (SUBDIAG_HEIGHT + 1),      // C
                           POSE_DIM, POSE_DIM,                       // rows, cols
                           FACTOR_WIDTH,                             // strideA
                           SUBDIAG_HEIGHT,                           // strideB
                           SUBDIAG_HEIGHT,                           // strideC
                           1, 1);                                    // alpha, beta
                // Add H_Pf
                matrix_add(H_factor + (LDMK_DIM + 1) * FACTOR_WIDTH - 1,  // A
                           C + (pose_idx + 1) * SUBDIAG_HEIGHT - 1,       // B
                           C + (pose_idx + 1) * SUBDIAG_HEIGHT - 1,       // C
                           1, POSE_DIM,                                   // rows, cols
                           FACTOR_WIDTH,                                  // strideA
                           SUBDIAG_HEIGHT,                                // strideB
                           SUBDIAG_HEIGHT,                                // strideC
                           1, 1);                                         // alpha, beta
            }
        }

        // Invert the block diagonal matrix
        invert_block_diagonal_matrix(A_chunk, LDMK_CHUNK * LDMK_DIM, LDMK_DIM);

        // Compute BA^-1 = B * A^-1 by computing (BA^-1)^T = A^-T * B^T
        matmul2(TOTAL_LDMK_DIM, TOTAL_POSE_DIM, TOTAL_LDMK_DIM,     // I, J, K
                A_chunk, B_chunk, B_Ainv_chunk, B_Ainv_chunk,       // A, B, D, C
                TOTAL_LDMK_DIM, SUBDIAG_HEIGHT,                     // strideA, strideB
                SUBDIAG_HEIGHT, SUBDIAG_HEIGHT,                     // strideD, strideC
                1, 1, 0,                                            // scaleA, scaleB, scaleD
                false, false); // transposeA, transposeB

        // Compute C = C - BA^-1 * B by computing C^T = C^T - B^T * (BA^-1)^T
        matmul2(TOTAL_POSE_DIM, TOTAL_POSE_DIM, TOTAL_LDMK_DIM,     // I, J, K
                B_chunk, B_Ainv_chunk, C, C,                        // A, B, D, C
                SUBDIAG_HEIGHT, SUBDIAG_HEIGHT,                     // strideA, strideB
                SUBDIAG_HEIGHT, SUBDIAG_HEIGHT,                     // strideD, strideC
                -1, 1, 1,                                           // scaleA, scaleB, scaleD
                true, true);                                        // transposeA, transposeB


    }

    cholesky(C, SUBDIAG_HEIGHT, SUBDIAG_HEIGHT);


    return 0;
}
