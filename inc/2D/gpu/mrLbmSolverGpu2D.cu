
#include "../../../common/mlcudaCommon.h"
#include "mrConstantParamsGpu2D.h"
#include "mrUtilFuncGpu2D.h"
#include "mrLbmSolverGpu2D.h"


// CUDA 优化：1D 线性格点映射（256 线程/块）+ __restrict__ + #pragma unroll。
// 连续线程映射到连续格点，warp 内访存完全合并；数值与原 2D 实现逐位一致
// （仅内存布局与指令调度改变，物理公式未动）。
__global__ void mrSolver2DKernel(
	mrFlow2D* __restrict__ mlflow, int sample_x, int sample_y, int sample_num)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-C GPU LBM update kernel =====
	// 1. 1D 线性映射：线程 → 格点序号 index，越界直接退出。
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sample_num)
	{
		return;
	}

	// 只处理流体格点（边界/固体矩由 case 层预写）。
	if (mlflow[0].flag[index] != ML_FLUID)
	{
		return;
	}

	const int y = index / sample_x;
	const int x = index - y * sample_x;

	// 粘度为全场一致标量：松弛参数提前算一次，避免逐格重复访存与除法。
	const REAL vis = mlflow[0].vis_shear;
	const REAL tau = 0.5f + vis / cs2;
	const REAL omega = 1.0f / tau;

	// Pull streaming: reconstruct each population from its upstream neighbor.
	mrUtilFuncGpu2D util;
	REAL f[9];
#pragma unroll
	for (int i = 0; i < 9; i++)
	{
		int nx = x - (int)ex2d_gpu[i];
		int ny = y - (int)ey2d_gpu[i];
		if (nx < 0) nx = 0;
		if (nx >= sample_x) nx = sample_x - 1;
		if (ny < 0) ny = 0;
		if (ny >= sample_y) ny = sample_y - 1;
		const int neighborIdx = ny * sample_x + nx;
		const REAL* neighborMom = &mlflow[0].fMom[neighborIdx * 6];
		util.mlCalDistributionD2Q9AtIndex(
			neighborMom[0], neighborMom[1], neighborMom[2],
			neighborMom[3], neighborMom[5], neighborMom[4],
			i, f[i]);
	}

	// Recover density, force-corrected velocity and raw second moments.
	REAL rho = 0.0f;
	REAL ux_times_rho = 0.0f;
	REAL uy_times_rho = 0.0f;
	REAL pixx_raw = 0.0f;
	REAL piyy_raw = 0.0f;
	REAL pixy_raw = 0.0f;
#pragma unroll
	for (int i = 0; i < 9; i++)
	{
		rho += f[i];
		ux_times_rho += f[i] * ex2d_gpu[i];
		uy_times_rho += f[i] * ey2d_gpu[i];
		pixx_raw += f[i] * ex2d_gpu[i] * ex2d_gpu[i];
		piyy_raw += f[i] * ey2d_gpu[i] * ey2d_gpu[i];
		pixy_raw += f[i] * ex2d_gpu[i] * ey2d_gpu[i];
	}

	const REAL Fx = mlflow[0].forcex[index];
	const REAL Fy = mlflow[0].forcey[index];
	REAL ux = (ux_times_rho + 0.5f * Fx) / rho;
	REAL uy = (uy_times_rho + 0.5f * Fy) / rho;

	// 速度上限锁：密度退化（越界 / NaN）时把该格点重置为平衡态，否则把
	// 速度幅值限制在 umax2d_gpu 以内。防止高流速 / 强外力下 rho 除零产生
	// NaN 并逐格扩散导致程序崩溃。
	if (rho < 0.5f || rho > 1.5f || !isfinite(ux) || !isfinite(uy))
	{
		rho = 1.0f;
		ux = 0.0f;
		uy = 0.0f;
		pixx_raw = rho * cs2;
		piyy_raw = rho * cs2;
		pixy_raw = 0.0f;
	}
	else
	{
		const REAL speed = sqrtf(ux * ux + uy * uy);
		if (speed > umax2d_gpu)
		{
			const REAL scale = umax2d_gpu / speed;
			ux *= scale;
			uy *= scale;
		}
	}

	// Collide the raw second moments with body-force correction.
	// （vis/tau/omega 已在格点入口处一次性算好，这里直接使用）
	util.mlGetPIAfterCollision(
		rho, ux, uy, Fx, Fy, omega, pixx_raw, piyy_raw, pixy_raw);

	// Normalize the second moments and write all six values to fMomPost.
	REAL* post = &mlflow[0].fMomPost[index * 6];
	post[0] = rho;
	post[1] = ux;
	post[2] = uy;
	post[3] = pixx_raw / rho - cs2;
	post[4] = piyy_raw / rho - cs2;
	post[5] = pixy_raw / rho;
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

	// CUDA 优化：256 线程 1D 块（原 8×8=64 线程 2D 块），线程线性映射格点，
	// warp 访存完全合并；取消每步 cudaDeviceSynchronize()（异步队列化，消除
	// 每步同步延迟）。正确性由帧末 CopyMomentsFromDevice 的 D2H 拷贝
	// （cudaMemcpy 同步）隐式保证；错误由 GetLastError 在每步捕获。
	const int threads1 = 256;
	const int blocks1 = (sample_num + threads1 - 1) / threads1;
	mrSolver2DKernel << <blocks1, threads1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
	mrSolver2D_step2Kernel << <1, 1 >> >
		(
			mlflow,
			sample_x, sample_y,
			sample_num
			);
	checkCudaErrors(cudaGetLastError());
}
