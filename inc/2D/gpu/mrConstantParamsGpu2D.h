#pragma once
#ifndef _MRCONSTANTPARAMSGPU2DH_
#define _MRCONSTANTPARAMSGPU2DH_


__constant__ float ex2d_gpu[9] = { 0,1,0,-1,0,1,-1,-1,1 };
__constant__ float ey2d_gpu[9] = { 0,0,1,0,-1,1,1,-1,-1 };
__constant__ int index2dInv_gpu[9] = { 0,3,4,1,2,7,8,5,6 };
__constant__ float w2d_gpu[9] = { 4.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };
__constant__ float as2 = 3.0f;
__constant__ float cs2 = 1.0f / 3.0f;
// 速度上限锁：流速幅值超过该值（格点单位）时被拉回上限。
// LBM 稳定性要求 |u| << cs（cs = sqrt(cs2) ≈ 0.577），取 0.3 既能覆盖
// UI 允许的最大入口速度 0.2 + 扰动，又能阻止高 Re / 强外力下的速度发散 NaN。
__constant__ float umax2d_gpu = 0.3f;


#endif // !_MRCONSTANTPARAMSGPU2DH_
