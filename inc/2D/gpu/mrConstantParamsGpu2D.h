#pragma once
#ifndef _MRCONSTANTPARAMSGPU2DH_
#define _MRCONSTANTPARAMSGPU2DH_


__constant__ float ex2d_gpu[9] = { 0,1,0,-1,0,1,-1,-1,1 };
__constant__ float ey2d_gpu[9] = { 0,0,1,0,-1,1,1,-1,-1 };
__constant__ int index2dInv_gpu[9] = { 0,3,4,1,2,7,8,5,6 };
__constant__ float w2d_gpu[9] = { 4.0 / 9.0, 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };
__constant__ float as2 = 3.0f;
__constant__ float cs2 = 1.0f / 3.0f;


#endif // !_MRCONSTANTPARAMSGPU2DH_
