#include "types.h"

Vector2f add_Vector2f(Vector2f v1, Vector2f v2, float scale) {
    Vector2f v;
    v.x = v1.x + scale * v2.x;
    v.y = v1.y + scale * v2.y;
    return v;
}

Vector3f add_Vector3f(Vector3f v1, Vector3f v2, float scale) {
    Vector3f v;
    v.x = v1.x + scale * v2.x;
    v.y = v1.y + scale * v2.y;
    v.z = v1.z + scale * v2.z;
    return v;
}

Quaternionf mult_Quaternionf(Quaternionf q1, Quaternionf q2) {
    Quaternionf q;
    q.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    q.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    q.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    q.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return q;
}

Quaternionf create_Quaternionf(float w, float x, float y, float z) {
    Quaternionf q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

Quaternionf Quaternionf_from_Vector3f(Vector3f v) {
    Quaternionf q;
    q.w = 0;
    q.x = v.x;
    q.y = v.y;
    q.z = v.z;
    return q;
}

Quaternionf conjugate_Quaternionf(Quaternionf q) {
    Quaternionf q_conj;
    q_conj.w = q.w;
    q_conj.x = -q.x;
    q_conj.y = -q.y;
    q_conj.z = -q.z;
    return q_conj;
}

Vector3f Vector3f_from_Quaternionf(Quaternionf q) {
    Vector3f v;
    v.x = q.x;
    v.y = q.y;
    v.z = q.z;
    return v;
}

Vector3f apply_rotation(Quaternionf q, Vector3f v) {
    Quaternionf v_quat = Quaternionf_from_Vector3f(v);
    Quaternionf qv = mult_Quaternionf(q, v_quat);
    Quaternionf q_conj = conjugate_Quaternionf(q);
    Quaternionf qv_rot = mult_Quaternionf(qv, q_conj);
    return Vector3f_from_Quaternionf(qv_rot);
}

Vector3f apply_transform(SE3 T, Vector3f v) {
    Vector3f v_rot = apply_rotation(T.q, v);
    return add_Vector3f(v_rot, T.t, 1);
}
