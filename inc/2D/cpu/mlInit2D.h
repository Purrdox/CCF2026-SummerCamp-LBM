#pragma once
#ifndef MRINIT2DH_
#define MRINIT2DH_
#include "mrFlow2D.h"
#include "mrConstantParamsCpu2D.h"
class mrInitHandler2D
{
public:
	mrInitHandler2D();
	~mrInitHandler2D();

	void mlInitBoundaryCpu(std::vector<mrFlow2D*>  mlflowvec, int scale, REAL L);
	void mlInitFlowVarCpu(std::vector<mrFlow2D*>  mlflowvec, int scale, REAL L);

private:

};

mrInitHandler2D::mrInitHandler2D()
{
}

mrInitHandler2D::~mrInitHandler2D()
{
}

inline void mrInitHandler2D::mlInitBoundaryCpu(std::vector<mrFlow2D*> mlflowvec, int scale, REAL L)
{
	int Nx = mlflowvec[scale]->param->samples.x;
	int Ny = mlflowvec[scale]->param->samples.y;
#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			int curind = y * mlflowvec[scale]->param->samples.x + x;

			if (mlflowvec[scale]->flag[curind] != ML_SOLID && mlflowvec[scale]->flag[curind] != ML_INVALID)
			{
				REAL _cellsize = mlflowvec[scale]->param->delta_x;
				REAL _endx;
				REAL _endy;

				REAL i, j;
				for (i = mlflowvec[scale]->param->start_pt.x; i < mlflowvec[scale]->param->domian_size.x; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endx = i - mlflowvec[scale]->param->delta_x;

				for (i = mlflowvec[scale]->param->start_pt.y; i < mlflowvec[scale]->param->domian_size.y; i += mlflowvec[scale]->param->delta_x)
				{
				}
				_endy = i - mlflowvec[scale]->param->delta_x;

				//mlflowvec[scale]->flag[curind] = ML_FLUID;
				if (x == 0)
					mlflowvec[scale]->flag[curind] = ML_WALL_LEFT;
				if (x == Nx - 1)
					mlflowvec[scale]->flag[curind] = ML_WALL_RIGHT;
				if (y == Ny - 1)
					mlflowvec[scale]->flag[curind] = ML_WALL_UP;
				if (y == 0)
					mlflowvec[scale]->flag[curind] = ML_WALL_DOWN;

				if (
				/*	(powf(x - 0.5 * Nx + 0.5, 2)
						)
					<= 0.06 * Nx * 0.06 * Nx
					&&*/
					y == 0
					)
				{
					mlflowvec[scale]->flag[curind] = ML_INLET;
				}

				if (
					(
						powf(x - 0.5 * Nx + 0.5, 2) +
						powf(y - 0.2 * Ny + 0.5, 2))
					<= 0.1 * Nx * 0.1 * Nx
					)
				{
					mlflowvec[scale]->flag[curind] = ML_SOLID;
				}

			}
		}
	}
}

inline void mrInitHandler2D::mlInitFlowVarCpu(std::vector<mrFlow2D*> mlflowvec, int scale, REAL L)
{

	//#pragma omp parallel  for num_threads(omp_get_num_threads() -1)
	for (long y = 0; y < mlflowvec[scale]->param->samples.y; y++)
	{
		for (long x = 0; x < mlflowvec[scale]->param->samples.x; x++)
		{
			int curind = y * mlflowvec[scale]->param->samples.x + x;
			REAL ux = 0.0;
			REAL uy = 0.2;
			REAL rho = 1.0;

			if (mlflowvec[scale]->flag[curind] == ML_INLET)// || mlflowvec[scale]->flag[curind] == ML_FLUID)
			{
				uy = 0.2;// uyPhy / mparam.u0p * mparam.labma;
			}
			if (mlflowvec[scale]->flag[curind] == ML_SOLID)// || mlflowvec[scale]->flag[curind] == ML_FLUID)
			{
				uy = 0.0;// uyPhy / mparam.u0p * mparam.labma;
			}
			REAL pop[9];
			REAL U2 = ux * ux + uy * uy;
			for (int i = 0; i < 9; i++)
			{
				REAL cu = ex2d_cpu[i] * ux + ey2d_cpu[i] * uy; // c k*u
				pop[i] = w2d_cpu[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U2);
			}
			REAL invRho = 1 / rho;
			REAL pixx = pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[8];
			REAL piyy = pop[2] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8];
			REAL pixy = pop[5] - pop[6] + pop[7] - pop[8];
			pixx = 1 * (pixx * invRho - cs2_cpu);
			piyy = 1 * (piyy * invRho - cs2_cpu);
			pixy = 1 * (pixy * invRho);
			mlflowvec[scale]->fMomPost[curind * 6 + 0] = mlflowvec[scale]->fMom[curind * 6 + 0] = rho;
			mlflowvec[scale]->fMomPost[curind * 6 + 1] = mlflowvec[scale]->fMom[curind * 6 + 1] = ux;
			mlflowvec[scale]->fMomPost[curind * 6 + 2] = mlflowvec[scale]->fMom[curind * 6 + 2] = uy;
			mlflowvec[scale]->fMomPost[curind * 6 + 3] = mlflowvec[scale]->fMom[curind * 6 + 3] = pixx;
			mlflowvec[scale]->fMomPost[curind * 6 + 4] = mlflowvec[scale]->fMom[curind * 6 + 4] = piyy;
			mlflowvec[scale]->fMomPost[curind * 6 + 5] = mlflowvec[scale]->fMom[curind * 6 + 5] = pixy;
		}
	}
}


#endif // !MRINIT2DH_
