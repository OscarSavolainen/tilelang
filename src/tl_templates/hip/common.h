// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include <ck_tile/core.hpp>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#define HIPRT_INF_F __int_as_float(0x7f800000)
#define HIPRT_NEGINF_F __int_as_float(0xff800000)
#define HIPRT_NAN_F __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F __int_as_float(0x80000000)
#define HIPRT_ZERO_F 0.0f
#define HIPRT_ONE_F 1.0f

/* double precision constants */
#define HIPRT_INF __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN __hiloint2double(0xfff80000, 0x00000000)

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__

#define half _Float16
#define __float2half_rn(x) half(x)

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

// Do we use fnuz or OCP for fp8
#ifndef USE_FP8_FNUZ
#define USE_FP8_FNUZ true
#endif

using float16_t = _Float16;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 =
    __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 =
    __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;

using half_t = float16_t;

using bfloat16_t = __hip_bfloat16;

struct bfloat16x2 {
  bfloat16_t data[2];
};

struct bfloat16x4 {
  bfloat16_t data[4];
};

struct bfloat16x8 {
  bfloat16_t data[8];
};

struct bfloat16x16 {
  bfloat16_t data[16];
};

typedef
    __attribute__((__vector_size__(4 * sizeof(short)))) short bfloat16x4_vec;

// TODO: add fp8 and 4xfp8 packing

// OCP FP8 currently not supported on MI300s: 
// https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/reference/low_fp_types.html 
#if USE_FP8_FNUZ
  // Fp8_e4m3_fnuz
  using fp8_e4_t = __hip_fp8_e4m3_fnuz;
  using fp8_e4_2_t = __hip_fp8x2_e4m3_fnuz;
  using fp8_e4_4_t = __hip_fp8x4_e4m3_fnuz;
  struct fp8_e4_8_t {
    fp8_e4_t data[8];
  };
  struct fp8_e4_16_t {
    fp8_e4_t data[16];
  };

  // Fp8_e5m3_fnuz
  using fp8_e5_t = __hip_fp8_e5m2_fnuz;
  using fp8_e5_2_t = __hip_fp8x2_e5m2_fnuz;
  using fp8_e5_4_t = __hip_fp8x4_e5m2_fnuz;
  struct fp8_e5_8_t {
    fp8_e5_t data[8];
  };
  struct fp8_e5_16_t {
    fp8_e5_t data[16];
  };
#else
  // USE OCP FP8
  using fp8_e4_t = __hip_fp8_e4m3;
  using fp8_e4_2_t = __hip_fp8x2_e4m3;
  using fp8_e4_4_t = __hip_fp8x4_e4m3;
  struct fp8_e4_8_t {
    fp8_e4_t data[8];
  };
  struct fp8_e4_16_t {
    fp8_e4_t data[16];
  };

  // Fp8_e5m2
  using fp8_e5_t = __hip_fp8_e5m2;
  using fp8_e5_2_t = __hip_fp8x2_e5m2;
  using fp8_e5_4_t = __hip_fp8x4_e5m2;
  struct fp8_e5_8_t {
    fp8_e5_t data[8];
  };
  struct fp8_e5_16_t {
    fp8_e5_t data[16];
  };
#endif

typedef
    __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t float8x8_vec;


// TL_DEVICE unsigned __pack_float_e4(const fp8_e4_t w, const fp8_e4_t x, const fp8_e4_t y, const fp8_e4_t z) {
//   unsigned v0 = *((unsigned int *)&w);
//   unsigned v1 = *((unsigned int *)&x);
//   unsigned v2 = *((unsigned int *)&y);
//   unsigned v3 = *((unsigned int *)&z);
//   return (v1 << 24) | (v1 << 16) | (v1 << 8) | v0;
// }

using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
