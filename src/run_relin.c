#include <math.h>


// Function to compute arccos using a series expansion
float acos_impl(float x) {
    if (x < -1.0 || x > 1.0) {
        return -1;  // Return an error for out of range inputs
    }

    // Arccos(x) = pi/2 - arcsin(x)
    // Approximate arcsin(x) using a series expansion
    float term = x;
    float sum = x;
    float xsq = x * x;
    float num = x;
    float denom = 1.0;
    int n;

    for (n = 1; n < 5; n++) {  // Limit the series to 10 terms for approximation
        num *= xsq * (2 * n - 1) * (2 * n - 1);
        denom *= (2 * n) * (2 * n + 1);
        term = num / denom;
        sum += term;
    }

    // pi/2 - arcsin(x)
    float pi_over_2 = 1.57079632679;  // Approximation of pi/2
    return pi_over_2 - sum;
}

// Sine function using Taylor series
float sin_impl(float x) {
    float sum = 0.0;
    float x_pow = x;
    int denom = 1;
    for (int n = 2; n <= 10; n += 2) {  // 10 terms of the series
        sum += x_pow / denom;

        x_pow *= (x * x);
        denom *= -(n * (n + 1));
    }
    return sum;
}

// Cosine function using Taylor series
float cos_impl(float x) {
    float sum = 0.0;
    float x_pow = 1;
    int denom = 1;
    for (int n = 1; n <= 9; n += 2) {  // 10 terms of the series
        sum += x_pow / denom;

        x_pow *= (x * x);
        denom *= -(n * (n + 1));
    }
    return sum;
}

void LogMap(const float Q[3][3], float H[3][3], float omega[3]) {
    const float R11 = Q[0][0], R12 = Q[0][1], R13 = Q[0][2];
    const float R21 = Q[1][0], R22 = Q[1][1], R23 = Q[1][2];
    const float R31 = Q[2][0], R32 = Q[2][1], R33 = Q[2][2];

    // Get trace(R)
    const float tr = R11 + R22 + R33;

    float magnitude;
    const float tr_3 = tr - 3.0; // could be non-negative if the matrix is off orthogonal
    if (tr_3 < -1e-6) {
      // this is the normal case -1 < trace < 3
      float theta = acos_impl((tr - 1.0) / 2.0);
      magnitude = theta / (2.0 * sin(theta));
    } else {
      // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
      // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
      // see https://github.com/borglab/gtsam/issues/746 for details
      magnitude = 0.5 - tr_3 / 12.0 + tr_3*tr_3/60.0;
    }

    omega[0] = magnitude * (R32 - R23);
    omega[1] = magnitude * (R13 - R31);
    omega[2] = magnitude * (R21 - R12);

    float theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];

    if(theta2 < 1e-5) {
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                H[i][j] = i == j? 1.0 : 0.0;
            }
        }
        return;
    }

    float theta = sqrt(theta2);

    float W[3][3] = {0};
    W[0][1] = -omega[2];
    W[0][2] = omega[1];
    W[1][2] = -omega[0];

    H[0][0] = 1;
    H[1][1] = 1;
    H[2][2] = 1;

    H[0][1] = -0.5 * omega[2];
    H[0][2] = 0.5 * omega[1];
    H[1][2] = -0.5 * omega[0];

    H[1][0] = 0.5 * omega[2];
    H[2][0] = -0.5 * omega[1];
    H[2][1] = 0.5 * omega[0];

    float c = 1 / theta2 - (1 + cos(theta)) / (2 * theta * sin(theta));

    float w00 = omega[0] * omega[0];
    float w01 = omega[0] * omega[1];
    float w02 = omega[0] * omega[2];
    float w11 = omega[1] * omega[1];
    float w12 = omega[1] * omega[2];
    float w22 = omega[1] * omega[2];

    H[0][0] += c * (-w22 + w11);
    H[0][1] += c * w01;
    H[0][2] += c * w02;
    H[1][0] += H[0][1];
    H[1][1] += c * (-w22 + w00);
    H[1][2] += c * w12;
    H[2][0] += H[2][0];
    H[2][1] += H[1][2];
    H[2][2] += c * (-w11 - w00);

}

void Local(const float measured_rotation[3][3], const float measured_translation[3], 
           const float hx_rotation[3][3], const float hx_translation[3], 
           const Hlocal[6][6]) {
    float DR[3][3];
    float omega[3];

    LogMap(hx_rotation, DR, omega);

    printf("%f %f %f\n\n", omega[0], omega[1], omega[2]);

    printf("%f %f %f\n", DR[0][0], DR[0][1], DR[0][2]);
    printf("%f %f %f\n", DR[1][0], DR[1][1], DR[1][2]);
    printf("%f %f %f\n", DR[2][0], DR[2][1], DR[2][2]);



}

int main() {
    for(int i = 0; i < 9; i++) {
        float Hlocal[6][6];
        float R1[3][3] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
        float R2[3][3] = {0.8660, 0.5, 0, -0.5, 0.8660, 0, 0, 0, 1};
        const float t1[3] = {0}, t2[3] = {0};

        Local(R1, t1, R2, t2, Hlocal);
    }


}

