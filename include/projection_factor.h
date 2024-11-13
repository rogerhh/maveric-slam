#ifndef PROJECTION_FACTOR_H
#define PROJECTION_FACTOR_H

#include "types.h"

typedef struct ProjectionFactor_t {
    // Each factor connects a landmark and a pose
    Vector3f* landmark;
    SE3* pose;

    // The measurement is 2-dimensional, and represents the x, y coordinates of 
    // the landmark projected onto the image plane
    Vector2f measurement;
    Vector2f error;

    // Camera intrinsics
    Camera camera;
} ProjectionFactor;

ProjectionFactor* create_ProjectionFactor(Vector3f* landmark, SE3* pose, 
                                          Vector2f measurement, Camera camera);



// Normalize a 3D point to a 2D point on the image plane
Vector2f project2d(const Vector3f trans_xyz);

// Project a 3D point to a 2D point on the image plane using the camera intrinsics
Vector2f cam_project(const Vector3f trans_xyz, const Camera camera);

void compute_error_ProjectionFactor(ProjectionFactor* factor);


#endif // PROJECTION_FACTOR_H
