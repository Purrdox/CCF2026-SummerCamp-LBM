
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"


__global__ void mrSolver2DKernel(
	mrFlow2D* mlflow, /*float* vx_dev,*/ int sample_x, int sample_y, int sample_num)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-C GPU LBM update kernel =====
	// Map the CUDA thread to a lattice node.
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= sample_x || y >= sample_y)
		return;

	int index = y * sample_x + x;

	// Process ML_FLUID nodes only.
	if (mlflow[0].flag[index] != ML_FLUID)
		return;

	// Pull streaming: reconstruct each D2Q9 population from its upstream
	// neighbor's six stored moments.
	REAL f[9];
	mrUtilFuncGpu2D util;
	for (int i = 0; i < 9; i++)
	{
		int srcX = x - (int)ex2d_gpu[i];
		int srcY = y - (int)ey2d_gpu[i];

		// Clamp to domain bounds.
		srcX = max(0, min(sample_x - 1, srcX));
		srcY = max(0, min(sample_y - 1, srcY));

		int srcIdx = srcY * sample_x + srcX;
		REAL srcRho   = mlflow[0].fMom[srcIdx * 6 + 0];
		REAL srcUx    = mlflow[0].fMom[srcIdx * 6 + 1];
		REAL srcUy    = mlflow[0].fMom[srcIdx * 6 + 2];
		REAL srcPixx  = mlflow[0].fMom[srcIdx * 6 + 3];
		REAL srcPiyy  = mlflow[0].fMom[srcIdx * 6 + 4];
		REAL srcPixy  = mlflow[0].fMom[srcIdx * 6 + 5];

		util.mlCalDistributionD2Q9AtIndex(
			srcRho, srcUx, srcUy,
			srcPixx, srcPixy, srcPiyy,
			i, f[i]);
	}

	// Recover density, first moments, and raw second moments from the
	// nine pulled populations.
	REAL rho = 0.0f;
	REAL jx  = 0.0f;
	REAL jy  = 0.0f;
	REAL rawPixx = 0.0f;
	REAL rawPiyy = 0.0f;
	REAL rawPixy = 0.0f;
	for (int i = 0; i < 9; i++)
	{
		REAL ei = f[i];
		rho += ei;
		jx  += ex2d_gpu[i] * ei;
		jy  += ey2d_gpu[i] * ei;
		rawPixx += ex2d_gpu[i] * ex2d_gpu[i] * ei;
		rawPiyy += ey2d_gpu[i] * ey2d_gpu[i] * ei;
		rawPixy += ex2d_gpu[i] * ey2d_gpu[i] * ei;
	}

	// Force-corrected macroscopic velocity.
	REAL fx = mlflow[0].forcex[index];
	REAL fy = mlflow[0].forcey[index];
	REAL ux = (jx + 0.5f * fx) / rho;
	REAL uy = (jy + 0.5f * fy) / rho;

	// Relaxation rate: omega = 1 / (3 * nu + 0.5) for D2Q9 with cs^2=1/3.
	REAL omega = 1.0f / (3.0f * mlflow[0].vis_shear + 0.5f);

	// BGK collision with body-force correction on raw second moments.
	util.mlGetPIAfterCollision(
		rho, ux, uy, fx, fy, omega,
		rawPixx, rawPiyy, rawPixy);

	// Normalize second moments and write to fMomPost.
	mlflow[0].fMomPost[index * 6 + 0] = rho;
	mlflow[0].fMomPost[index * 6 + 1] = ux;
	mlflow[0].fMomPost[index * 6 + 2] = uy;
	mlflow[0].fMomPost[index * 6 + 3] = rawPixx / rho;
	mlflow[0].fMomPost[index * 6 + 4] = rawPiyy / rho;
	mlflow[0].fMomPost[index * 6 + 5] = rawPixy / rho;
	// ===== ASSIGNMENT FILL END: P1-C =====
}
__host__ __device__
void MomSwap(REAL*& pt1, REAL*& pt2) {
	REAL* temp = pt1;
	pt1 = pt2;
	pt2 = temp;
}
__global__ void mrSolver2D_step2Kernel(
	mrFlow2D* mlflow, int sample_x, int sample_y, int sample_num)
{
	MomSwap(mlflow[0].fMom, mlflow[0].fMomPost);
}


void mrSolver2DGpu(mrFlow2D* mlflow, MLFluidParam2D* param)
{
	int sample_x = param->samples.x;
	int sample_y = param->samples.y;
	int sample_num = sample_x * sample_y;

	dim3 threads1(BLOCK_NX, BLOCK_NY, 1);
	dim3 grid1(
		ceil(REAL(sample_x) / threads1.x),
		ceil(REAL(sample_y) / threads1.y), 1
	);
	mrSolver2DKernel << <grid1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	mrSolver2D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
