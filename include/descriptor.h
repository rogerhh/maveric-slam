#pragma once

#include <stdint.h>
#include <math.h>

#define DESCRIPTOR_SIZE 256

// Compute the distance between two descriptors as the 2 norm
float descriptor_distance(const float* descriptor0, const float* descriptor1) {
  float distance = 0;
  for(int i = 0; i < DESCRIPTOR_SIZE; i++) {
    distance += descriptor0[i] * descriptor1[i];
  }
  return distance;
}
