#pragma once

void normalize_points(const int num_points, const float points[][2], 
                      const float K[3][3], float normalized_points[][2]);

void compute_essential_matrix(const int num_points, 
                              const float pts1_norm[][2], 
                              const float pts2_norm[][2], 
                              float E[3][3]);

float compute_reprojection_error(const float point1[2], 
                                 const float point2[2], 
                                 const float E[3][3]);

void ransac_essential_matrix(const int num_points,
                             const float points1[][2], const float points2[][2], 
                             const float K[3][3],
                             const int num_iterations, const float inlier_threshold,
                             float best_E[3][3], int *best_inliers, int *num_inliers);

void recover_pose_from_essential_matrix(float E[3][3],float R1[3][3], 
                                        float R2[3][3], float t[3]);
