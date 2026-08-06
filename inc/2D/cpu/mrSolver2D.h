#pragma once
#ifndef MRSOLVER2DH_
#define MRSOLVER2DH_

#include "../gpu/mrLbmSolverGpu2D.h"

#include "../../../common/colorramp.h"
#include "mlInit2D.h"

using namespace Mfree;
using namespace std;

class mrSolver2D
{
public:
	mrSolver2D();
	~mrSolver2D();
	void AttachLbmHost(std::vector<mrFlow2D*> lbmvec);
	void AttachLbmDevice(std::vector<mrFlow2D*> lbmvec_dev);
	void AttachMapping(MLMappingParam& mapping);
	void mlInit();
	void mlIterateGpu();
	void mlTransData2Host(int i);
	void mlTransData2Gpu();
	void mlDeepCopy(mrFlow2D* mllbm_host, mrFlow2D* mllbm_dev, int i);
	void mlVisVelocitySlice(long upw, long uph, int scaleNum, int frame);
	void mlSavePPM(const char* filename, float* data, int mWidth, int mHeight);
	void mlSaveFmom(const char* filename, int frame);
public:

	std::vector<mrFlow2D*> lbmvec;
	std::vector<mrFlow2D*> lbm_dev_gpu;
	MLMappingParam mparam;
	float L;
	int gpuId = 0;

	// rigid var
	//float* vx_dev;
	//float* vx_cpu;
	//checkCudaErrors(cudaMalloc(&vx_dev, 2 * sizeof(REAL)));
	//checkCudaErrors(_MLCuMemcpy(vx_dev, vx_cpu, 2 * sizeof(REAL), cudaMemcpyHostToDevice));

private:
	mrInitHandler2D mlinithandler2d;

};

mrSolver2D::mrSolver2D()
{
}

mrSolver2D::~mrSolver2D()
{
}

inline void mrSolver2D::AttachLbmHost(std::vector<mrFlow2D*> lbmvec)
{
	this->lbmvec = lbmvec;
}

inline void mrSolver2D::AttachLbmDevice(std::vector<mrFlow2D*> lbmvec_dev)
{
	this->lbm_dev_gpu = lbmvec_dev;
}



inline void mrSolver2D::AttachMapping(MLMappingParam& mapping)
{
	this->mparam.l0p = mapping.l0p;
	this->mparam.N = mapping.N;

	this->mparam.u0p = mapping.u0p;
	this->mparam.labma = mapping.labma;

	this->mparam.tp = mapping.tp;
	this->mparam.roup = mapping.roup;

}



inline void mrSolver2D::mlInit()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		mlinithandler2d.mlInitBoundaryCpu(lbmvec, i, L);
		mlinithandler2d.mlInitFlowVarCpu(lbmvec, i, L);
	}
}

inline void mrSolver2D::mlIterateGpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		mrSolver2DGpu(lbm_dev_gpu[i], lbmvec[i]->param);

	}
	//rigidbody dt 
	
}

inline void mrSolver2D::mlTransData2Host(int i)
{
	mrFlow2D* mllbm_host = new mrFlow2D();
	checkCudaErrors(cudaSetDevice(gpuId));
	checkCudaErrors(_MLCuMemcpy(mllbm_host, lbm_dev_gpu[i], 1 * sizeof(mrFlow2D), cudaMemcpyDeviceToHost));
	checkCudaErrors(_MLCuMemcpy(lbmvec[i]->fMom, (mllbm_host->fMom), lbmvec[i]->count * 6 * sizeof(REAL), cudaMemcpyDeviceToHost));

}

inline void mrSolver2D::mlTransData2Gpu()
{
	checkCudaErrors(cudaSetDevice(gpuId));
	for (int i = 0; i < lbmvec.size(); i++)
	{
		if (lbm_dev_gpu[i] != NULL)
		{
			checkCudaErrors(cudaFree(lbm_dev_gpu[i]));
		}
		_MLCuMalloc((void**)&lbm_dev_gpu[i], sizeof(mrFlow2D));
		mlDeepCopy(lbmvec[i], lbm_dev_gpu[i], i);
	}
}

inline void mrSolver2D::mlDeepCopy(mrFlow2D* mllbm_host, mrFlow2D* mllbm_dev, int i)
{

	float* fMom_dev;
	float* fMomPost_dev;
	MLLATTICENODE_FLAG* flag_dev;
	MLFluidParam3D* param_dev;
	float* forcex_dev;
	float* forcey_dev;


#pragma region MallocData

	checkCudaErrors(cudaMalloc(&fMom_dev, 6 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&fMomPost_dev, 6 * mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&flag_dev, mllbm_host->count * sizeof(MLLATTICENODE_FLAG)));
	checkCudaErrors(cudaMalloc(&param_dev, 1 * sizeof(MLFluidParam3D)));
	checkCudaErrors(cudaMalloc(&forcex_dev, mllbm_host->count * sizeof(REAL)));
	checkCudaErrors(cudaMalloc(&forcey_dev, mllbm_host->count * sizeof(REAL)));


#pragma endregion

#pragma region MEMCPY

	checkCudaErrors(_MLCuMemcpy(fMom_dev, mllbm_host->fMom, 6 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(fMomPost_dev, mllbm_host->fMomPost, 6 * mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(flag_dev, mllbm_host->flag, mllbm_host->count * sizeof(MLLATTICENODE_FLAG), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(param_dev, mllbm_host->param, 1 * sizeof(MLFluidParam3D), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcex_dev, mllbm_host->forcex, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(forcey_dev, mllbm_host->forcey, mllbm_host->count * sizeof(REAL), cudaMemcpyHostToDevice));


#pragma endregion
	checkCudaErrors(_MLCuMemcpy(mllbm_dev, mllbm_host, 1 * sizeof(mrFlow2D), cudaMemcpyHostToDevice));

#pragma region DeepCOPY

	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMom), &fMom_dev, sizeof(fMom_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->fMomPost), &fMomPost_dev, sizeof(fMomPost_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->flag), &flag_dev, sizeof(flag_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->param), &param_dev, sizeof(param_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcex), &forcex_dev, sizeof(forcex_dev), cudaMemcpyHostToDevice));
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->forcey), &forcey_dev, sizeof(forcey_dev), cudaMemcpyHostToDevice));

#pragma endregion

	// §1.4 功能 4 预留：smoke 设备端指针置 NULL（方案 B）。
	// 功能 4 的烟雾平流在 host 端执行，设备端不做分配；若不显式置 NULL，
	// 上面的 sizeof(mrFlow2D) 整结构体拷贝会把 host 堆指针拷进设备结构体，
	// 届时对 deviceCopy.smoke 做 cudaFree 会对 host 指针操作 → invalid device
	// pointer 崩溃。kernel 不读 smoke，NULL 无害。
	REAL* smoke_dev = NULL;
	checkCudaErrors(_MLCuMemcpy(&(mllbm_dev->smoke), &smoke_dev, sizeof(smoke_dev), cudaMemcpyHostToDevice));
}

inline void mrSolver2D::mlVisVelocitySlice(long upw, long uph, int scaleNum, int frame)
{
	int upnum = 0;
	int baseScale = 0;
	float* cutslice_ve = new float[lbmvec[0]->param->samples.x * lbmvec[0]->param->samples.y];
	int num = 0;
	for (int i = 0; i < scaleNum; i++)
	{
		int stx = 0;
		int sty = 0;
		int stz = 0;
		int edx = 0;
		int edy = 0;
		int edz = 0;
		stx = 0;
		sty = 0;
		stz = 0;
		edx = lbmvec[i]->param->samples.x;
		edy = lbmvec[i]->param->samples.y;
		for (int y = sty; y < edy; y++)
			for (int x = stx; x < edx; x++)
			{
				int curind = y * lbmvec[i]->param->samples.x + x;

				/*ucurind.ux = lbmvec[i]->ux[curind] / mparam.labma * mparam.u0p;
				ucurind.uy = lbmvec[i]->uy[curind] / mparam.labma * mparam.u0p;
				ucurind.uz = lbmvec[i]->uz[curind] / mparam.labma * mparam.u0p;*/
				cutslice_ve[num] = sqrtf(
					lbmvec[i]->fMom[6 * curind + 1] * lbmvec[i]->fMom[6 * curind + 1] +
					lbmvec[i]->fMom[6 * curind + 2] * lbmvec[i]->fMom[6 * curind + 2]);


				num++;
			}
	}
	float* vertices = new float[upw * uph * 3];
	num = 0;


	ColorRamp color_m;
	for (int j = uph - 1; j >= 0; j--)
	{
		for (int i = 0; i < upw; i++)
		{
			float x = (float)i / ((float)upw) * lbmvec[0]->param->samples.x;
			float y = (float)j / ((float)uph) * lbmvec[0]->param->samples.y;
			int x00 = floor(x);
			int x01 = x00 + 1;
			int y00 = floor(y);
			int y01 = y00 + 1;

			float rateX = x - x00;
			float rateY = y - y00;
			if (x00 < 0) x00 = 0;
			if (x01 >= lbmvec[0]->param->samples.x) x01 = lbmvec[0]->param->samples.x - 1;
			if (y00 < 0) y00 = 0;
			if (y01 >= lbmvec[0]->param->samples.y) y01 = lbmvec[0]->param->samples.y - 1;

			int ind0 = x00 + y00 * lbmvec[0]->param->samples.x;
			int ind1 = x01 + y00 * lbmvec[0]->param->samples.x;
			int ind2 = x00 + y01 * lbmvec[0]->param->samples.x;
			int ind3 = x01 + y01 * lbmvec[0]->param->samples.x;
			double vv = (1 - rateY) * ((1 - rateX) * cutslice_ve[ind0] + rateX * cutslice_ve[ind1]) +
				(rateY) * ((1 - rateX) * cutslice_ve[ind2] + rateX * cutslice_ve[ind3]);
			vv = vv / 0.32;
			vec3 color(0, 0, 0);
			color_m.set_GLcolor(vv, COLOR__MAGMA, color, false);
			vertices[num++] = color.x;
			vertices[num++] = color.y;
			vertices[num++] = color.z;
		}
	}

	char filename[2048];
	sprintf(filename, "../dataMR2D/ppm_ve/im%05d.ppm", frame);
	mlSavePPM(filename, vertices, upw, uph);
	delete[] cutslice_ve;
	delete[] vertices;
}

inline void mrSolver2D::mlSavePPM(const char* filename, float* data, int mWidth, int mHeight)
{
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << mWidth << " " << mHeight << "\n255\n";
	for (unsigned i = 0; i < mWidth * mHeight * 3; ++i) {
		ofs << (unsigned char)(data[i] * 255);
	}
	ofs.close();
}

inline void mrSolver2D::mlSaveFmom(const char* filename, int frame)
{
	FILE* fp = fopen(filename, "wb");

	/*int Nx = lbmvec[0]->param->samples.x;
	int Ny = lbmvec[0]->param->samples.y;
	fwrite(&Nx, sizeof(float), 1, fp);
	fwrite(&Ny, sizeof(float), 1, fp);*/
	fwrite(lbmvec[0]->fMom, sizeof(float), lbmvec[0]->count * 6, fp);
	fclose(fp);
}

#endif // !MRSOLVER2DH_