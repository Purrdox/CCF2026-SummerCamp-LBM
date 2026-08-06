#pragma once
#ifndef MRCONSTANTPARAMSCPU2DH_
#define MRCONSTANTPARAMSCPU2DH_
const float ex2d_cpu[9] = { 0,1,0,-1,0,1,-1,-1,1 };
const float ey2d_cpu[9] = { 0,0,1,0,-1,1,1,-1,-1 };
const int index2dInv_cpu[9] = { 0,3,4,1,2,7,8,5,6 };
const float w2d_cpu[9] = { 4.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };
const float as2_cpu = 3.0;
const float cs2_cpu = 1.0 / 3.0;
// 速度上限锁（CPU 端）：与 GPU 端 umax2d_gpu 保持一致，
// 用于钳制边界/出口外推等 CPU 写入的流速，防止其放大进流场。
const float umax2d_cpu = 0.3f;
#endif // !MRCONSTANTPARAMSCPU3DH_
