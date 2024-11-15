#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define GEMMINI_TYPE float
#define elem_t GEMMINI_TYPE
#define scale_t GEMMINI_TYPE

// Perform C += A_scale_factor * A^(transpose_A * T) * B_scale_factor * B^(transpose_B * T)
// Assume A, B, C are stored in row-major order
void matmul(
  size_t dim_I, size_t dim_J, size_t dim_K, 
  const elem_t* A, const elem_t* B, elem_t* C,
  size_t stride_A, size_t stride_B, size_t stride_C,
  scale_t A_scale_factor, scale_t B_scale_factor,
  bool transpose_A, bool transpose_B) {
  size_t stride_Ai, stride_Ak, stride_Bk, stride_Bj, stride_Ci = stride_C, stride_Cj = 1;
  if(transpose_A) {
    stride_Ai = 1;
    stride_Ak = stride_A;
  }
  else {
    stride_Ai = stride_A;
    stride_Ak = 1;
  }
  if(transpose_B) {
    stride_Bk = 1;
    stride_Bj = stride_B;
  }
  else {
    stride_Bk = stride_B;
    stride_Bj = 1;
  }
  const elem_t* Ai = A;
  elem_t* Ci = C;
  for(size_t i = 0; i < dim_I; i++) {
    const elem_t* Bj = B;
    elem_t* Cij = Ci;
    for(size_t j = 0; j < dim_J; j++) {
      const elem_t* Aik = Ai;
      const elem_t* Bkj = Bj;
      for(size_t k = 0; k < dim_K; k++) {
        *Cij += A_scale_factor * (*Aik) * B_scale_factor * (*Bkj);
        Aik += stride_Ak;
        Bkj += stride_Bk;
      }
      Bj += stride_Bj;
      Cij += stride_Cj;
    }
    Ai += stride_Ai;
    Ci += stride_Ci;
  }
}

// Perform C = A_scale_factor * A^(transpose_A * T) * B_scale_factor * B^(transpose_B * T) + D_scale_factor * D
// Assume A, B, C, D are stored in row-major order
void matmul2(
  size_t dim_I, size_t dim_J, size_t dim_K, 
  const elem_t* A, const elem_t* B, 
  const elem_t* D, elem_t* C,
  size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
  scale_t A_scale_factor, scale_t B_scale_factor, scale_t D_scale_factor,
  bool transpose_A, bool transpose_B) {
    size_t stride_Ai, stride_Ak, stride_Bk, stride_Bj;
    size_t stride_Ci = stride_C, stride_Cj = 1;
    size_t stride_Di = stride_D, stride_Dj = 1;

    if(transpose_A) {
        stride_Ai = 1;
        stride_Ak = stride_A;
    } 
    else {
        stride_Ai = stride_A;
        stride_Ak = 1;
    }
    if(transpose_B) {
        stride_Bk = 1;
        stride_Bj = stride_B;
    } 
    else {
        stride_Bk = stride_B;
        stride_Bj = 1;
    }

    if(D) {
        // Set C = D_scale_factor * D
        const elem_t* Di = D;
        elem_t* Ci = C;
        for(size_t i = 0; i < dim_I; i++) {
            const elem_t* Dij = Di;
            elem_t* Cij = Ci;
            for(size_t j = 0; j < dim_J; j++) {
                *Cij = D_scale_factor * (*Dij);
                Dij += stride_Dj;
                Cij += stride_Cj;
            }
            Di += stride_Di;
            Ci += stride_Ci;
        }
    }

    const elem_t* Ai = A;
    elem_t* Ci = C;
    for(size_t i = 0; i < dim_I; i++) {
        const elem_t* Bj = B;
        elem_t* Cij = Ci;
        for(size_t j = 0; j < dim_J; j++) {
            const elem_t* Aik = Ai;
            const elem_t* Bkj = Bj;
            for(size_t k = 0; k < dim_K; k++) {
                *Cij += A_scale_factor * (*Aik) * B_scale_factor * (*Bkj);
                Aik += stride_Ak;
                Bkj += stride_Bk;
            }
            Bj += stride_Bj;
            Cij += stride_Cj;
        }
        Ai += stride_Ai;
        Ci += stride_Ci;
    }
}
