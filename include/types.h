#ifndef TYPES_H
#define TYPES_H

typedef struct Vector2f_t {
    float x, y;
} Vector2f;

typedef struct Vector3f_t {
    float x, y, z;
} Vector3f;

typedef struct Quaternionf_t {
    float w, x, y, z;
} Quaternionf;

typedef struct SE3_t {
    Quaternionf q;
    Vector3f t;
} SE3;

typedef struct Camera_t {
    float fx, fy, cx, cy;
} Camera;

Vector2f add_Vector2f(Vector2f v1, Vector2f v2, float scale);
Vector3f add_Vector3f(Vector3f v1, Vector3f v2, float scale);
Quaternionf mult_Quaternionf(Quaternionf q1, Quaternionf q2);

Quaternionf create_Quaternionf(float w, float x, float y, float z);
Quaternionf Quaternionf_from_Vector3f(Vector3f v);
Quaternionf conjugate_Quaternionf(Quaternionf q);
Vector3f Vector3f_from_Quaternionf(Quaternionf q);
Vector3f apply_rotation(Quaternionf q, Vector3f v);
Vector3f apply_transform(SE3 T, Vector3f v);

#endif // TYPES_H
