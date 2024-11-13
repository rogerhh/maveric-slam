#include "projection_factor.h"

ProjectionFactor* create_ProjectionFactor(Vector3f* landmark, SE3* pose, Vector2f measurement, Camera camera) {
    ProjectionFactor* factor = (ProjectionFactor*) malloc(sizeof(ProjectionFactor));
    factor->landmark = landmark;
    factor->pose = pose;
    factor->measurement = measurement;
    factor->camera = camera;
    return factor;
}

Vector2f project2d(const Vector3f trans_xyz) {
    Vector2f proj;
    proj.x = trans_xyz.x / trans_xyz.z;
    proj.y = trans_xyz.y / trans_xyz.z;
    return proj;
}

Vector2f cam_project(const Vector3f trans_xyz, const Camera camera) {
    Vector2f proj = project2d(trans_xyz);
    Vector2f res;
    res.x = proj.x * camera.fx + camera.cx;
    res.y = proj.y * camera.fy + camera.cy;
    return res;
}

void compute_error_ProjectionFactor(ProjectionFactor* factor) {
    const Vector3f landmark = *(factor->landmark);
    const SE3 pose = *(factor->pose);
    Vector3f landmark_in_pose = apply_transform(pose, landmark);
    Vector2f projected = cam_project(landmark_in_pose, factor->camera);
    factor->error = add_Vector2f(projected, factor->measurement, -1);
}
