#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "svd.h"
#include "pnp_solver.h"

void normalize_points(const int num_points, const float points[][2], 
                      const float K[3][3], float normalized_points[][2]) {
    for (int i = 0; i < num_points; ++i) {
        normalized_points[i][0] = (points[i][0] - K[0][2]) / K[0][0];
        normalized_points[i][1] = (points[i][1] - K[1][2]) / K[1][1];
    }
}

void compute_essential_matrix(const int num_points, 
                              const float points1_norm[][2], 
                              const float points2_norm[][2], 
                              float E[3][3]) {

    // This function needs to be implemented. Here's a placeholder for now.
    // Populate E with some values for testing purposes.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            E[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

float compute_reprojection_error(const float point1[2], 
                                 const float point2[2], 
                                 const float E[3][3]) {
    float point1_h[3] = {point1[0], point1[1], 1.0};
    float point2_h[3] = {point2[0], point2[1], 1.0};
    
    // E * pt1
    float temp[3];
    for (int i = 0; i < 3; ++i) {
        temp[i] = E[i][0] * point1_h[0] + E[i][1] * point1_h[1] + E[i][2] * point1_h[2];
    }
    
    // pt2^T * temp
    float error = fabs(point2_h[0] * temp[0] + point2_h[1] * temp[1] + point2_h[2] * temp[2]);
    
    return error;
}

#define MAX_NUM_INLIERS 1000

// RANSAC function to estimate the essential matrix and find inliers
void ransac_essential_matrix(const int num_points,
                             const float points1[][2], const float points2[][2], 
                             const float K[3][3],
                             const int num_iterations, const float inlier_threshold,
                             float best_E[3][3], int *best_inliers, int *num_inliers) {
    int max_inliers = 0;
    float temp_E[3][3];
    int inliers[MAX_NUM_INLIERS];
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Randomly select 8 points
        int indices[8];
        for (int i = 0; i < 8; ++i) {
            indices[i] = rand() % num_points;
        }
        
        // Normalize the selected points
        float points1_sample[8][2], points2_sample[8][2];
        for (int i = 0; i < 8; ++i) {
            points1_sample[i][0] = points1[indices[i]][0];
            points1_sample[i][1] = points1[indices[i]][1];
            points2_sample[i][0] = points2[indices[i]][0];
            points2_sample[i][1] = points2[indices[i]][1];
        }
        
        float points1_norm[8][2], points2_norm[8][2];
        normalize_points(8, points1_sample, K, points1_norm);
        normalize_points(8, points2_sample, K, points2_norm);
        
        // Compute the essential matrix from the selected points
        compute_essential_matrix(8, points1_norm, points2_norm, temp_E);
        
        // Count the number of inliers
        int inlier_count = 0;
        for (int i = 0; i < num_points; ++i) {
            float error = compute_reprojection_error(points1[i], points2[i], temp_E);
            if (error < inlier_threshold && inlier_count < MAX_NUM_INLIERS) {
                inliers[inlier_count++] = i;
            }
        }
        
        // Update the best model if we have more inliers
        if (inlier_count > max_inliers || inlier_count == MAX_NUM_INLIERS) {
            max_inliers = inlier_count;
            *num_inliers = inlier_count;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    best_E[i][j] = temp_E[i][j];
                }
            }
            for (int i = 0; i < inlier_count; ++i) {
                best_inliers[i] = inliers[i];
            }
        }
    }
}

// Dummy implementation of SVD (replace with actual implementation)
void call_svd(float A[3][3], float U[3][3], float S[3], float Vt[3][3]) {
  svd(A[0][0], A[0][1], A[0][2], 
      A[1][0], A[1][1], A[1][2], 
      A[2][0], A[2][1], A[2][2], 
      U[0][0], U[0][1], U[0][2], 
      U[1][0], U[1][1], U[1][2], 
      U[2][0], U[2][1], U[2][2], 
      S[0], S[1], S[2], Vt[0][0], 
      Vt[0][1], Vt[0][2], Vt[1][0], 
      Vt[1][1], Vt[1][2], Vt[2][0], Vt[2][1], Vt[2][2]);
}

// Recover rotation and translation from an essential matrix
void recover_pose_from_essential_matrix(float E[3][3],float R1[3][3], 
                                        float R2[3][3], float t[3]) {

    // Perform SVD on the essential matrix
    float U[3][3], S[3], Vt[3][3];
    svd(E, U, S, Vt);
    
    // Enforce the rank-2 constraint on the essential matrix
    S[2] = 0.0;
    
    // Define the matrix W and its transpose
    float W[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
    float Wt[3][3] = {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}};
    
    // Compute the two possible rotation matrices
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R1[i][j] = U[i][0] * W[0][j] + U[i][1] * W[1][j] + U[i][2] * W[2][j];
            R2[i][j] = U[i][0] * Wt[0][j] + U[i][1] * Wt[1][j] + U[i][2] * Wt[2][j];
        }
    }
    
    // Compute the translation vector
    for (int i = 0; i < 3; ++i) {
        t[i] = U[i][2];
    }
}

