#pragma once
#ifndef MRFLOW2DH_
#define MRFLOW2DH_

#include "../../../common/mlFluid.h"
#include "../../../common/mlFluidParam.h"
#include "../../../common/mlLbmCommon.h"
#include "../../../common/mlCuRunTime.h"
//#include "../../../3rdParty/helper_cuda.h"
#include "../../../3rdParty/helper_cuda.h"
#include "../../../common/mlCommon.h"

#include <fstream>
#include <iostream>
#include <vector>

class mrFlow2D
{
public:
	mrFlow2D();
	~mrFlow2D();
	MLLATTICENODE_FLAG* flag;// domain flag

/*
* rhoVar 0
* uxVar  1
* uyVar  2
* pixx   3
* piyy   4
* pixy   5
*/
	REAL* fMom;
	REAL* fMomPost;
	MLFluidParam2D* param;
	REAL* forcex;
	REAL* forcey;
	REAL vis_shear;
	cudaStream_t stream;
	long count = 0;
	REAL* vis_velocity;
	REAL* smoke;   // §1.4 功能 4 预留：烟雾密度场（host 平流，设备端仅保持结构一致）

	void Create
	(
		REAL x0, REAL y0,
		long width, long height,
		REAL delta_x,
		REAL box_w, REAL box_h,
		REAL vis, REAL gy
	);
private:

};

inline mrFlow2D::mrFlow2D()
{
}

inline mrFlow2D::~mrFlow2D()
{
}

inline void mrFlow2D::Create(
	REAL x0, REAL y0,
	long width, long height,
	REAL delta_x,
	REAL box_w, REAL box_h,
	REAL vis, REAL gy
)
{
	this->vis_shear = vis;
	param = new MLFluidParam2D[1];
	long sample_x_count = 0; long sample_y_count = 0;
	REAL endx = 0; REAL endy = 0;
	REAL i = 0;
	for (i = x0; i < box_w + x0; i += delta_x)
	{
		sample_x_count++;
	}
	endx = i - delta_x;
	for (i = y0; i < box_h + y0; i += delta_x)
	{
		sample_y_count++;
	}
	endy = i - delta_x;
	count = sample_x_count * sample_y_count;
	param->start_pt.x = x0;		param->start_pt.y = y0;
	param->end_pt.x = endx;		param->end_pt.y = endy;
	param->delta_x = delta_x;	param->delta_t = delta_x;
	param->validCount = count;	param->box_size.x = box_w; 	param->box_size.y = box_h;
	param->domian_size.x = width;		param->domian_size.y = height;
	param->samples.x = sample_x_count;	param->samples.y = sample_y_count;
	param->gx = 0;
	param->gy = gy;
	fMom = new REAL[6 * count];
	fMomPost = new REAL[6 * count];
	flag = new MLLATTICENODE_FLAG[count];
	forcex = new REAL[count];
	forcey = new REAL[count];
	smoke = new REAL[count];   // §1.4 功能 4 预留

	for (long y = 0; y < sample_y_count; y++)
	{
		for (long x = 0; x < sample_x_count; x++)
		{
			int num = y * sample_x_count + x;
			flag[num] = ML_FLUID;
			forcex[num] = 0;
			forcey[num] = 0;
			smoke[num] = 0;
		}
	}
}


#endif // !MRFLOW2DH_
