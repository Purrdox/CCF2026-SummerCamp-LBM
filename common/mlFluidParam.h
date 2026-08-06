#pragma once
#ifndef _MLFLUIDPARAM_
#define _MLFLUIDPARAM_
#include "mlCoreWin.h"
#include "mlvertex.h"
#include "mlDataType.h"
struct MLFluidParam3D
{
	mlVertex3f start_pt; //the start position of the simulation domain in 2D world coordinate system
	mlVertex3f end_pt; //the end position of the simulation domain in 2D world coordinate system
	GVLSize3l samples;     //the sample numbers in each dimensions
	GVLSize3l domian_size;     //the domain size
	GVLSize3f box_size;     //the domain size

	mlVertex3f smoke_start_pt;
	mlVertex3f smoke_end_pt;
	GVLSize3f smoke_size; //the smoke size
	REAL delta_x;
	REAL delta_t;
	int scaleNum;
	REAL vis_shear;
	REAL vis_bulk;
	int validCount;

	size_t NUMBER_GHOST_FACE_XY;
	size_t NUMBER_GHOST_FACE_XZ;
	size_t NUMBER_GHOST_FACE_YZ;
	size_t NUM_BLOCK_X;
	size_t NUM_BLOCK_Y;
	size_t NUM_BLOCK_Z;


	REAL gx;
	REAL gy;
	REAL gz;
};

struct MLPstuct2D
{
	REAL ux0;
	REAL uy0;
	REAL rho0;
	REAL gx;
	REAL gy;

	REAL rho_gas;
	REAL rho_liquid;
	REAL phi_gas;
	REAL phi_liquid;
	REAL vis_shear_gas;
	REAL vis_shear_liquid;

	REAL sigma;
	REAL kapa;
	REAL beta;

	REAL interfaceWidth;
	REAL M;
	REAL vis_phi;

	REAL vis_phi0;
	REAL vis_phi1_2;
	REAL vis_phi3;
	REAL vis_phi6_7;
	REAL vis_phi8;

	REAL phiaverage;
	REAL ratio;

	REAL liquidP;
	REAL liquidR;

	REAL bndP;
	REAL bndR;
	REAL cd;
	REAL cdphi;

	REAL scaleTime;
	bool upwind;


	REAL G;
	REAL theta;

	mlVertex2f sphereCenter;
	REAL sphereRadius;
};

struct MLPstuct3D
{

	REAL ux0;
	REAL uy0;
	REAL uz0;
	REAL rho0;
	REAL gx;
	REAL gy;
	REAL gz;

	REAL rho_gas;
	REAL rho_liquid;
	REAL phi_gas;
	REAL phi_liquid;
	REAL vis_shear_gas;
	REAL vis_shear_liquid;

	REAL sigma;
	REAL kapa;
	REAL beta;

	REAL interfaceWidth;
	REAL M;
	REAL vis_phi;

	REAL vis_phi0;
	REAL vis_phi1_2;
	REAL vis_phi3;
	REAL vis_phi6_7;
	REAL vis_phi8;

	REAL phiaverage;
	REAL ratio;

	REAL liquidP;
	REAL liquidR;

	REAL bndP;
	REAL bndR;
	REAL cd;
	REAL cdphi;

	REAL scaleTime;
	bool upwind;


	REAL G;
	REAL theta;
	REAL theta_in;
};

struct MLDensiyuParam
{
	mlVertex3f start_pt;
	mlVertex3f end_pt;
	GVLSize3l samples;     //the sample numbers in each dimensions
	GVLSize3l domian_size;     //the domain size
	GVLSize3f box_size;     //the domain size
	REAL delta_x;
	REAL delta_t;
	int validCount;

};
struct MLFluidParam2D
{
	mlVertex2f start_pt; //the start position of the simulation domain in 2D world coordinate system
	mlVertex2f end_pt; //the end position of the simulation domain in 2D world coordinate system
	GVLSize2l samples;     //the sample numbers in each dimensions
	GVLSize2l domian_size;     //the domain size
	GVLSize2f box_size;     //the domain size
								//mlVertex2f smoke_start_pt;
							//mlVertex2f smoke_end_pt;
							//GVLSize2f smoke_size; //the smoke size
	REAL delta_x;
	REAL delta_t;

	int scaleNum;
	REAL vis_shear;
	REAL vis_bulk;
	int validCount;
	REAL gx;
	REAL gy;
};

struct mlNsFluidParam
{

	mlVertex3f macoffU;
	mlVertex3f macoffV;
	mlVertex3f macoffW;
	mlVertex3f macoffD;

	int width;
	int height;
	int depth;
	REAL hx;
	REAL dt;
	REAL density;
	int uCount;
	int vCount;
	int wCount;
	int dCount;
	REAL timeStep;
	REAL viscosity;
	GVLSize3l u_size;
	GVLSize3l v_size;
	GVLSize3l w_size;
	GVLSize3l d_size;
};
struct HybridMappingParm
{
	REAL rho0;
	REAL rhol0;
	REAL P0;
	REAL delta_x;
	REAL delta_t;
	REAL T0;

public:
	HybridMappingParm()
	{}
	HybridMappingParm
	(
		REAL _rho0,
		REAL _rhol0,
		REAL _P0,
		REAL _delta_x,
		REAL _delta_t,
		REAL _T0
	)
	{
		rho0 = _rho0;
		rhol0 = _rhol0;
		P0 = _P0;
		delta_x = _delta_x;
		delta_t = _delta_t;
		T0 = _T0;
	}
};

 
struct MLMappingParam
{
	REAL lp, tp, xp;
	REAL t0p, l0p;
	REAL N;
	REAL u0p;
	REAL viscosity_p;
	REAL viscosity_k;
	REAL labma;
	REAL roup;
public:
	MLMappingParam()
	{}
	MLMappingParam
	(
		REAL _uop,
		REAL _labma,
		REAL _l0p,
		REAL _N,
		REAL _roup
	)
	{
		u0p = _uop;
		labma = _labma;
		l0p = _l0p;
		N = _N;
		roup = _roup;
	}
};

#endif // !_MLFLUIDPARAM_
