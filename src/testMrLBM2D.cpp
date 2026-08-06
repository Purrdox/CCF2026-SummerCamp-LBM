#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "LbmCases2D.h"
#include "ParamsIO.h"
#include "../inc/2D/cpu/mrSolver2D.h"

#include <windows.h>
#include <windowsx.h>
#include <gl/GL.h>


// Dear ImGui（§5.1）：1.90.9，OpenGL2 + Win32 后端
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl2.h"

// imgui_impl_win32.h 将 WndProcHandler 声明置于 #if 0 块内（避免头文件依赖 windows.h），
// 需按注释自行前置声明（§5.1 第 5 条）
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
	HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#pragma comment(lib, "opengl32.lib")

namespace
{
	const int kTimerMilliseconds = 16;
	const int kPanelWidth = 350;
	const int kMinimumStepsPerFrame = 1;
	const int kMaximumStepsPerFrame = 50;
	const float kMaximumTargetLead = 1.75f;
	const float kMaximumWallSpeed = 0.25f;   // 请求 5：提高物体拖动/移动速度上限
	const int kInverseDirection[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

	// 功能 4（colorful 视图）：力幅度上限
	const float kFmax = 0.02f;            // §5.3 力幅度上限（与 blowStrength 滑块上限一致）

	struct Moments2D
	{
		REAL rho;
		REAL ux;
		REAL uy;
		REAL sxx;
		REAL syy;
		REAL sxy;
	};

	struct RectF
	{
		float left;
		float top;
		float right;
		float bottom;

		float Width() const
		{
			return right - left;
		}

		float Height() const
		{
			return bottom - top;
		}
	};

	struct UiLayout
	{
		RectF field;
		RectF panel;
	};

	float ClampFloat(float value, float minValue, float maxValue)
	{
		return std::max(minValue, std::min(value, maxValue));
	}

	bool IsFinite(REAL value)
	{
		return std::isfinite((double)value) != 0;
	}

	Moments2D ReadMoments(const mrFlow2D* flow, int index)
	{
		Moments2D value;
		value.rho = flow->fMom[index * 6 + 0];
		value.ux = flow->fMom[index * 6 + 1];
		value.uy = flow->fMom[index * 6 + 2];
		value.sxx = flow->fMom[index * 6 + 3];
		value.syy = flow->fMom[index * 6 + 4];
		value.sxy = flow->fMom[index * 6 + 5];
		return value;
	}

	bool AreMomentsUsable(const Moments2D& value)
	{
		return IsFinite(value.rho) &&
			IsFinite(value.ux) &&
			IsFinite(value.uy) &&
			IsFinite(value.sxx) &&
			IsFinite(value.syy) &&
			IsFinite(value.sxy) &&
			value.rho > 0.5f &&
			value.rho < 1.5f;
	}

	void ReleaseDeviceData(mrSolver2D& solver)
	{
		checkCudaErrors(cudaSetDevice(solver.gpuId));
		for (int i = 0; i < (int)solver.lbm_dev_gpu.size(); i++)
		{
			if (solver.lbm_dev_gpu[i] == NULL)
			{
				continue;
			}

			mrFlow2D deviceCopy;
			checkCudaErrors(_MLCuMemcpy(
				&deviceCopy,
				solver.lbm_dev_gpu[i],
				sizeof(mrFlow2D),
				cudaMemcpyDeviceToHost));

			if (deviceCopy.fMom != NULL) cudaFree(deviceCopy.fMom);
			if (deviceCopy.fMomPost != NULL) cudaFree(deviceCopy.fMomPost);
			if (deviceCopy.flag != NULL) cudaFree(deviceCopy.flag);
			if (deviceCopy.param != NULL) cudaFree(deviceCopy.param);
			if (deviceCopy.forcex != NULL) cudaFree(deviceCopy.forcex);
			if (deviceCopy.forcey != NULL) cudaFree(deviceCopy.forcey);

			checkCudaErrors(cudaFree(solver.lbm_dev_gpu[i]));
			solver.lbm_dev_gpu[i] = NULL;
		}
	}

	void CopyMomentsFromDevice(mrSolver2D& solver, int flowIndex)
	{
		if (flowIndex < 0 ||
			flowIndex >= (int)solver.lbmvec.size() ||
			flowIndex >= (int)solver.lbm_dev_gpu.size() ||
			solver.lbmvec[flowIndex] == NULL ||
			solver.lbm_dev_gpu[flowIndex] == NULL)
		{
			return;
		}

		mrFlow2D deviceCopy;
		mrFlow2D* host = solver.lbmvec[flowIndex];
		checkCudaErrors(cudaSetDevice(solver.gpuId));
		checkCudaErrors(_MLCuMemcpy(
			&deviceCopy,
			solver.lbm_dev_gpu[flowIndex],
			sizeof(mrFlow2D),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(_MLCuMemcpy(
			host->fMom,
			deviceCopy.fMom,
			host->count * 6 * sizeof(REAL),
			cudaMemcpyDeviceToHost));
	}

	// §4.4 粘度热更新：把 host 标量成员同步到设备结构体（仅热更新路径需要；
	// mlTransData2Gpu 重建路径会整体 deep copy）
	void SyncFlowScalarsToDevice(mrSolver2D& solver, int flowIndex)
	{
		if (flowIndex < 0 ||
			flowIndex >= (int)solver.lbmvec.size() ||
			flowIndex >= (int)solver.lbm_dev_gpu.size() ||
			solver.lbmvec[flowIndex] == NULL ||
			solver.lbm_dev_gpu[flowIndex] == NULL)
		{
			return;
		}

		mrFlow2D* host = solver.lbmvec[flowIndex];
		checkCudaErrors(cudaSetDevice(solver.gpuId));
		checkCudaErrors(_MLCuMemcpy(
			&solver.lbm_dev_gpu[flowIndex]->vis_shear,
			&host->vis_shear,
			sizeof(REAL),
			cudaMemcpyHostToDevice));
	}

	void SyncCellsToDevice(
		mrSolver2D& solver,
		int flowIndex,
		const std::vector<int>& indices,
		bool syncFlags)
	{
		if (flowIndex < 0 ||
			flowIndex >= (int)solver.lbmvec.size() ||
			flowIndex >= (int)solver.lbm_dev_gpu.size() ||
			solver.lbmvec[flowIndex] == NULL ||
			solver.lbm_dev_gpu[flowIndex] == NULL)
		{
			return;
		}

		mrFlow2D* host = solver.lbmvec[flowIndex];
		mrFlow2D deviceCopy;
		checkCudaErrors(cudaSetDevice(solver.gpuId));
		checkCudaErrors(_MLCuMemcpy(
			&deviceCopy,
			solver.lbm_dev_gpu[flowIndex],
			sizeof(mrFlow2D),
			cudaMemcpyDeviceToHost));

		if (syncFlags)
		{
			checkCudaErrors(_MLCuMemcpy(
				deviceCopy.flag,
				host->flag,
				host->count * sizeof(MLLATTICENODE_FLAG),
				cudaMemcpyHostToDevice));
		}

		if (indices.empty())
		{
			return;
		}

		std::vector<int> sorted = indices;
		std::sort(sorted.begin(), sorted.end());
		sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

		int rangeStart = -1;
		int previous = -2;
		for (int item = 0; item <= (int)sorted.size(); item++)
		{
			const int index = item < (int)sorted.size() ? sorted[item] : -2;
			const bool valid = index >= 0 && index < host->count;
			if (rangeStart < 0)
			{
				if (valid)
				{
					rangeStart = index;
					previous = index;
				}
				continue;
			}

			if (valid && index == previous + 1)
			{
				previous = index;
				continue;
			}

			const int count = previous - rangeStart + 1;
			checkCudaErrors(_MLCuMemcpy(
				deviceCopy.fMom + 6 * rangeStart,
				host->fMom + 6 * rangeStart,
				count * 6 * sizeof(REAL),
				cudaMemcpyHostToDevice));
			checkCudaErrors(_MLCuMemcpy(
				deviceCopy.fMomPost + 6 * rangeStart,
				host->fMomPost + 6 * rangeStart,
				count * 6 * sizeof(REAL),
				cudaMemcpyHostToDevice));
			checkCudaErrors(_MLCuMemcpy(
				deviceCopy.forcex + rangeStart,
				host->forcex + rangeStart,
				count * sizeof(REAL),
				cudaMemcpyHostToDevice));
			checkCudaErrors(_MLCuMemcpy(
				deviceCopy.forcey + rangeStart,
				host->forcey + rangeStart,
				count * sizeof(REAL),
				cudaMemcpyHostToDevice));

			rangeStart = valid ? index : -1;
			previous = valid ? index : -2;
		}
	}

	void SyncOuterBoundaryMomentsToDevice(
		mrSolver2D& solver,
		int flowIndex)
	{
		if (flowIndex < 0 ||
			flowIndex >= (int)solver.lbmvec.size() ||
			flowIndex >= (int)solver.lbm_dev_gpu.size() ||
			solver.lbmvec[flowIndex] == NULL ||
			solver.lbm_dev_gpu[flowIndex] == NULL)
		{
			return;
		}

		mrFlow2D* host = solver.lbmvec[flowIndex];
		mrFlow2D deviceCopy;
		checkCudaErrors(cudaSetDevice(solver.gpuId));
		checkCudaErrors(_MLCuMemcpy(
			&deviceCopy,
			solver.lbm_dev_gpu[flowIndex],
			sizeof(mrFlow2D),
			cudaMemcpyDeviceToHost));

		const int nx = (int)host->param->samples.x;
		const int ny = (int)host->param->samples.y;
		const size_t rowBytes = nx * 6 * sizeof(REAL);
		const size_t columnBytes = 6 * sizeof(REAL);

		checkCudaErrors(_MLCuMemcpy(
			deviceCopy.fMom,
			host->fMom,
			rowBytes,
			cudaMemcpyHostToDevice));
		checkCudaErrors(_MLCuMemcpy(
			deviceCopy.fMomPost,
			host->fMomPost,
			rowBytes,
			cudaMemcpyHostToDevice));
		checkCudaErrors(_MLCuMemcpy(
			deviceCopy.fMom + (ny - 1) * nx * 6,
			host->fMom + (ny - 1) * nx * 6,
			rowBytes,
			cudaMemcpyHostToDevice));
		checkCudaErrors(_MLCuMemcpy(
			deviceCopy.fMomPost + (ny - 1) * nx * 6,
			host->fMomPost + (ny - 1) * nx * 6,
			rowBytes,
			cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy2D(
			deviceCopy.fMom,
			rowBytes,
			host->fMom,
			rowBytes,
			columnBytes,
			ny,
			cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy2D(
			deviceCopy.fMomPost,
			rowBytes,
			host->fMomPost,
			rowBytes,
			columnBytes,
			ny,
			cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy2D(
			deviceCopy.fMom + (nx - 1) * 6,
			rowBytes,
			host->fMom + (nx - 1) * 6,
			rowBytes,
			columnBytes,
			ny,
			cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy2D(
			deviceCopy.fMomPost + (nx - 1) * 6,
			rowBytes,
			host->fMomPost + (nx - 1) * 6,
			rowBytes,
			columnBytes,
			ny,
			cudaMemcpyHostToDevice));
	}

	bool AverageNeighborMoments(
		const mrFlow2D* flow,
		int x,
		int y,
		const std::vector<MLLATTICENODE_FLAG>& oldFlags,
		const std::vector<MLLATTICENODE_FLAG>& newFlags,
		bool unchangedFluidOnly,
		Moments2D& result)
	{
		const int nx = (int)flow->param->samples.x;
		const int ny = (int)flow->param->samples.y;
		Moments2D sum = {};
		int count = 0;

		for (int dy = -1; dy <= 1; dy++)
		{
			for (int dx = -1; dx <= 1; dx++)
			{
				if (dx == 0 && dy == 0)
				{
					continue;
				}

				const int neighborX = x + dx;
				const int neighborY = y + dy;
				if (neighborX < 0 || neighborX >= nx ||
					neighborY < 0 || neighborY >= ny)
				{
					continue;
				}

				const int neighbor = neighborY * nx + neighborX;
				if (newFlags[neighbor] != ML_FLUID ||
					(unchangedFluidOnly && oldFlags[neighbor] != ML_FLUID))
				{
					continue;
				}

				const Moments2D value = ReadMoments(flow, neighbor);
				if (!AreMomentsUsable(value))
				{
					continue;
				}

				sum.rho += value.rho;
				sum.ux += value.ux;
				sum.uy += value.uy;
				sum.sxx += value.sxx;
				sum.syy += value.syy;
				sum.sxy += value.sxy;
				count++;
			}
		}

		if (count == 0)
		{
			return false;
		}

		const REAL inverseCount = 1.0f / (REAL)count;
		result.rho = sum.rho * inverseCount;
		result.ux = sum.ux * inverseCount;
		result.uy = sum.uy * inverseCount;
		result.sxx = sum.sxx * inverseCount;
		result.syy = sum.syy * inverseCount;
		result.sxy = sum.sxy * inverseCount;
		return true;
	}

	REAL ReconstructDistribution(const Moments2D& value, int direction)
	{
		const REAL a0 = value.rho;
		const REAL ax = value.ux * a0;
		const REAL ay = value.uy * a0;
		const REAL axx = value.rho * value.sxx;
		const REAL ayy = value.rho * value.syy;
		const REAL axy = value.rho * value.sxy;
		const REAL axxy =
			-2.0f * value.rho * value.uy * value.ux * value.ux +
			2.0f * axy * value.ux +
			axx * value.uy;
		const REAL axyy =
			-2.0f * value.rho * value.ux * value.uy * value.uy +
			2.0f * axy * value.uy +
			ayy * value.ux;

		REAL polynomial = 0.0f;
		switch (direction)
		{
		case 0:
			polynomial = a0 - 1.5f * axx - 1.5f * ayy;
			break;
		case 1:
			polynomial = a0 + 3.0f * ax + 3.0f * axx -
				4.5f * axyy - 1.5f * ayy;
			break;
		case 2:
			polynomial = a0 - 1.5f * axx - 4.5f * axxy +
				3.0f * ay + 3.0f * ayy;
			break;
		case 3:
			polynomial = a0 - 3.0f * ax + 3.0f * axx +
				4.5f * axyy - 1.5f * ayy;
			break;
		case 4:
			polynomial = a0 - 1.5f * axx + 4.5f * axxy -
				3.0f * ay + 3.0f * ayy;
			break;
		case 5:
			polynomial = a0 + 3.0f * ax + 3.0f * axx +
				9.0f * axy + 9.0f * axxy + 9.0f * axyy +
				3.0f * ay + 3.0f * ayy;
			break;
		case 6:
			polynomial = a0 - 3.0f * ax + 3.0f * axx -
				9.0f * axy + 9.0f * axxy - 9.0f * axyy +
				3.0f * ay + 3.0f * ayy;
			break;
		case 7:
			polynomial = a0 - 3.0f * ax + 3.0f * axx +
				9.0f * axy - 9.0f * axxy - 9.0f * axyy -
				3.0f * ay + 3.0f * ayy;
			break;
		case 8:
			polynomial = a0 + 3.0f * ax + 3.0f * axx -
				9.0f * axy - 9.0f * axxy + 9.0f * axyy -
				3.0f * ay + 3.0f * ayy;
			break;
		}

		return std::max((REAL)0.0f, w2d_cpu[direction] * polynomial);
	}

	void MapVorticityColor(float normalized, float& r, float& g, float& b)
	{
		const float centerR = 0.93f;
		const float centerG = 0.94f;
		const float centerB = 0.91f;
		const float amount = std::min(1.0f, fabsf(normalized));
		if (normalized < 0.0f)
		{
			r = centerR + amount * (0.12f - centerR);
			g = centerG + amount * (0.35f - centerG);
			b = centerB + amount * (0.72f - centerB);
		}
		else
		{
			r = centerR + amount * (0.78f - centerR);
			g = centerG + amount * (0.16f - centerG);
			b = centerB + amount * (0.18f - centerB);
		}
	}

	// 功能 4：HSV → RGB（h∈[0,1)，s/v∈[0,1]）
	void HsvToRgb(float h, float s, float v, float& r, float& g, float& b)
	{
		const int sector = (int)floorf(h * 6.0f) % 6;
		const float f = h * 6.0f - floorf(h * 6.0f);
		const float p = v * (1.0f - s);
		const float q = v * (1.0f - s * f);
		const float t = v * (1.0f - s * (1.0f - f));
		switch (sector)
		{
		case 0: r = v; g = t; b = p; break;
		case 1: r = q; g = v; b = p; break;
		case 2: r = p; g = v; b = t; break;
		case 3: r = p; g = q; b = v; break;
		case 4: r = t; g = p; b = v; break;
		default: r = v; g = p; b = q; break;
		}
	}

	// 功能 4：rho/ux/uy → RGB 流场着色（§6.2 修正）。
	// 平静（|u|≈0）→ 纯白；流速增大 → 按速度方向取色相 + 时间戳旋转，
	// 生成鲜艳的动态色；rho 微调明度（压缩区更亮）。
	// 数值范围参考：rho≈1.0（界 0.5~1.5），|u| 0~0.15（Karman 入口 0.10 /
	// Jet 入口 0.08，speedColorMax 即归一化上限）。
	// 关键修正（功能 4 第 2 点）：相对速度 = 实际速度 - 实时入口速度（refUx/refUy）。
	// 参考速度取自当前 ML_INLET 格点的实际 ux/uy（含扰动，随帧更新，非固定配置值）；
	// 当全场稳定同向流动时各格点速度≈入口速度 → 相对速度≈0 → 全白。
	// colorfulSaturation 控制颜色鲜艳度（0=全白，越高越鲜艳）。
	void MapFlowFieldColor(
		REAL rho,
		REAL ux,
		REAL uy,
		float refUx,
		float refUy,
		float speedColorMax,
		float colorfulSaturation,
		float timeSeconds,
		float& r,
		float& g,
		float& b)
	{
		const float effUx = (float)ux - refUx;
		const float effUy = (float)uy - refUy;
		const float speed = sqrtf(effUx * effUx + effUy * effUy);
		const float strength = ClampFloat(
			speed / std::max(speedColorMax, 1.0e-4f), 0.0f, 1.0f);
		// smoothstep：低速仍保持白底，向鲜艳色平滑过渡；再乘鲜艳度（0→全白）
		float blend = strength * strength * (3.0f - 2.0f * strength);
		blend *= ClampFloat(colorfulSaturation, 0.0f, 1.0f);

		const float kPi = 3.14159265f;
		// 方向色相 + 时间旋转（约 60s 一圈，变化缓慢，仅作轻微动感）
		float hue = atan2f(effUy, effUx) / (2.0f * kPi) + 0.5f
			+ 0.0167f * timeSeconds;
		hue -= floorf(hue);
		// rho 微调明度：rho=1.0 → 1.0，波动 ±0.02 → 明度 ±0.2
		const float value = ClampFloat(
			1.0f + ((float)rho - 1.0f) * 10.0f, 0.6f, 1.0f);

		float vr = 1.0f, vg = 1.0f, vb = 1.0f;
		HsvToRgb(hue, 1.0f, value, vr, vg, vb);
		// 与白色按 blend 混合：平静纯白，快速鲜艳
		r = 1.0f * (1.0f - blend) + vr * blend;
		g = 1.0f * (1.0f - blend) + vg * blend;
		b = 1.0f * (1.0f - blend) + vb * blend;
	}

	void BuildFieldImage(
		const mrFlow2D* flow,
		const DemoCaseDefinition& definition,
		DemoFieldView view,
		std::vector<unsigned char>& pixels)
	{
		const int nx = definition.nx;
		const int ny = definition.ny;
		std::vector<float> values(nx * ny, 0.0f);
		pixels.resize(nx * ny * 3);

		for (int y = 0; y < ny; y++)
		{
			for (int x = 0; x < nx; x++)
			{
				const int index = y * nx + x;
				if (view == DemoFieldView::VelocityMagnitude)
				{
					const REAL ux = flow->fMom[index * 6 + 1];
					const REAL uy = flow->fMom[index * 6 + 2];
					values[index] = sqrtf(ux * ux + uy * uy);
					continue;
				}
				if (view == DemoFieldView::Colorful)
				{
					continue;   // colorful 视图：不需要标量场，直接按 rho/ux/uy 上色
				}

				const int left = y * nx + std::max(0, x - 1);
				const int right = y * nx + std::min(nx - 1, x + 1);
				const int down = std::max(0, y - 1) * nx + x;
				const int up = std::min(ny - 1, y + 1) * nx + x;
				const REAL centerUx = flow->fMom[index * 6 + 1];
				const REAL centerUy = flow->fMom[index * 6 + 2];
				const REAL leftUy = flow->flag[left] == ML_SOLID
					? centerUy
					: flow->fMom[left * 6 + 2];
				const REAL rightUy = flow->flag[right] == ML_SOLID
					? centerUy
					: flow->fMom[right * 6 + 2];
				const REAL downUx = flow->flag[down] == ML_SOLID
					? centerUx
					: flow->fMom[down * 6 + 1];
				const REAL upUx = flow->flag[up] == ML_SOLID
					? centerUx
					: flow->fMom[up * 6 + 1];
				values[index] = 0.5f * ((rightUy - leftUy) - (upUx - downUx));
			}
		}

		ColorRamp colorRamp;
		// 功能 4：时间戳（ms）→ 秒，供 Colorful 视图色相旋转（GetTickCount 掩码防浮点精度损失）
		const float timeSeconds = (float)(GetTickCount() & 0x3FFFFFFF) * 0.001f;
		// 功能 4 第 2 点：colorful 视图实时参考速度 = 当前入口（ML_INLET）格点实际速度的均值。
		// 全场稳定同向流动时格点速度≈入口速度 → 相对速度≈0 → 全白。
		// 无入口格点（理论上不会发生）时回退到配置的 inletUx/inletUy。
		float refUx = (float)definition.inletUx;
		float refUy = (float)definition.inletUy;
		if (view == DemoFieldView::Colorful)
		{
			double sumUx = 0.0;
			double sumUy = 0.0;
			int inletCount = 0;
			for (int i = 0; i < nx * ny; i++)
			{
				if (flow->flag[i] == ML_INLET)
				{
					sumUx += flow->fMom[i * 6 + 1];
					sumUy += flow->fMom[i * 6 + 2];
					inletCount++;
				}
			}
			if (inletCount > 0)
			{
				refUx = (float)(sumUx / inletCount);
				refUy = (float)(sumUy / inletCount);
			}
		}

		for (int y = 0; y < ny; y++)
		{
			for (int x = 0; x < nx; x++)
			{
				const int index = y * nx + x;
				float r = 0.0f;
				float g = 0.0f;
				float b = 0.0f;

				if (flow->flag[index] == ML_SOLID)
				{
					r = 0.035f;
					g = 0.045f;
					b = 0.055f;
				}
				else if (flow->flag[index] == ML_WALL ||
					flow->flag[index] == ML_WALL_LEFT ||
					flow->flag[index] == ML_WALL_RIGHT ||
					flow->flag[index] == ML_WALL_DOWN ||
					flow->flag[index] == ML_WALL_UP)
				{
					r = 0.16f;
					g = 0.18f;
					b = 0.20f;
				}
				else if (view == DemoFieldView::Colorful)
				{
					// 功能 4：rho/ux/uy → RGB 流场着色（平静=白，快速=鲜艳+时间旋转）
					MapFlowFieldColor(
						flow->fMom[index * 6 + 0],
						flow->fMom[index * 6 + 1],
						flow->fMom[index * 6 + 2],
						refUx,
						refUy,
						definition.speedColorMax,
						definition.colorfulSaturation,
						timeSeconds,
						r, g, b);
				}
				else if (view == DemoFieldView::VelocityMagnitude)
				{
					const float normalized = ClampFloat(
						values[index] / definition.speedColorMax, 0.0f, 1.0f);
					vec3 color(0.0f, 0.0f, 0.0f);
					colorRamp.set_GLcolor(normalized, COLOR__MAGMA, color, false);
					r = color.x;
					g = color.y;
					b = color.z;
				}
				else
				{
					const float normalized = ClampFloat(
						values[index] / definition.vorticityColorMax, -1.0f, 1.0f);
					MapVorticityColor(normalized, r, g, b);
				}

				pixels[index * 3 + 0] = (unsigned char)(ClampFloat(r, 0.0f, 1.0f) * 255.0f);
				pixels[index * 3 + 1] = (unsigned char)(ClampFloat(g, 0.0f, 1.0f) * 255.0f);
				pixels[index * 3 + 2] = (unsigned char)(ClampFloat(b, 0.0f, 1.0f) * 255.0f);
			}
		}
	}

	// 请求 1：形状感知的物体重叠判定（全局单一形状；组合 A 约束：不旋转、轴对齐）。
	// 圆：欧氏距离 < 半径和；方形（Box）：两轴方向均重叠才算碰撞；
	// 菱形（Diamond）：L1 距离（|dx|+|dy|）< 半径和即碰撞。
	bool BodiesOverlap(
		float ax, float ay, float ar,
		float bx, float by, float br,
		ObstacleShape shape)
	{
		const float dx = fabsf(ax - bx);
		const float dy = fabsf(ay - by);
		switch (shape)
		{
		case ObstacleShape::Box:
			return dx < ar + br && dy < ar + br;
		case ObstacleShape::Diamond:
			return dx + dy < ar + br;
		case ObstacleShape::Circle:
		default:
			return dx * dx + dy * dy < (ar + br) * (ar + br);
		}
	}

	// 功能 4：LbmApp 内联成员函数（ApplyMouseEffects 等）调用的自由函数前置声明，
	// 定义在本文件下方（§5.4 面板豁免 / layout 现算）
	UiLayout CurrentLayout(HWND window);
	bool IsOverUiPanel(int mx, int my, const UiLayout& layout);
	bool UiWantsCaptureMouse();

	struct LbmApp
	{
		mrSolver2D solver;
		mrFlow2D* flow = NULL;
		std::vector<mrFlow2D*> flows;
		std::vector<mrFlow2D*> deviceFlows;
		std::vector<unsigned char> pixels;
		DemoCaseDefinition definition;
		DemoCaseId caseId = DemoCaseId::KarmanVortex;
		DemoFieldView fieldView = DemoFieldView::Vorticity;
		int frame = 0;
		int iteration = 0;
		int stepsPerFrame = 5;

		// 物体系统（§1.2）：多物体数组，bodyCount >= 1 表示存在物体
		std::array<RigidBody, kMaxBodies> bodies = {};
		int bodyCount = 0;
		int selectedBody = 0;
		ObstacleShape obstacleShape = ObstacleShape::Circle;
		REAL bodyForceX[kMaxBodies] = { 0 };
		REAL bodyForceY[kMaxBodies] = { 0 };
		REAL bodyTorque[kMaxBodies] = { 0 };
		bool hasForceSample[kMaxBodies] = { false };

		// 鼠标输入（§1.3）
		int mouseScreenX = 0;
		int mouseScreenY = 0;
		float mouseFieldX = 0.0f;
		float mouseFieldY = 0.0f;
		bool dragging = false;
		bool lButtonDown = false;
		bool rButtonDown = false;
		bool mButtonDown = false;
		float grabOffsetX = 0.0f;
		float grabOffsetY = 0.0f;

		// 功能 4（colorful 交互）
		enum class InteractionTool { None, Blow, Vortex };
		InteractionTool activeTool = InteractionTool::None;
		float brushRadius = 6.0f;
		float blowStrength = 0.004f;

		// 功能 4（colorful，§2 新增成员）
		HWND appWindow = NULL;               // 窗口句柄：hover 豁免需现算 layout（§5.4）
		int prevMouseScreenX = 0;            // 帧间鼠标差分 → Blow 方向（§5.5）
		int prevMouseScreenY = 0;
		std::vector<int> injectedForceCells; // 本帧注入力的格点集合（§5.2/§5.3）
		float blowDirX = 0.0f;               // 本帧吹风方向（屏幕差分 → 格点方向，§5.5）
		float blowDirY = 1.0f;

		// 功能 5 预留（Ctrl 调试 + 放大，§1.4）
		bool ctrlHeld = false;
		bool debugMode = false;
		int debugCellX = -1;
		int debugCellY = -1;
		bool magnifierEnabled = false;

		bool moveLeft = false;
		bool moveRight = false;
		bool moveUp = false;
		bool moveDown = false;
		bool paused = false;
		bool initialized = false;

		void Release()
		{
			ReleaseDeviceData(solver);
			if (flow != NULL)
			{
				delete[] flow->fMom;
				delete[] flow->fMomPost;
				delete[] flow->flag;
				delete[] flow->param;
				delete[] flow->forcex;
				delete[] flow->forcey;
				delete flow;
			}

			flow = NULL;
			flows.clear();
			deviceFlows.clear();
			solver.lbmvec.clear();
			solver.lbm_dev_gpu.clear();
			pixels.clear();
			initialized = false;
		}

		void Init(DemoCaseId newCase)
		{
			caseId = newCase;
			definition = GetDefaultDefinition(caseId);
			fieldView = definition.defaultView;
			stepsPerFrame = definition.initialStepsPerFrame;
			frame = 0;
			iteration = 0;
			paused = false;
			ClearMoveKeys();

			// 物体系统复位（§1.2 / §7.2）
			bodies.fill({});
			obstacleShape = ObstacleShape::Circle;
			bodyCount = definition.hasMovableObstacle ? 1 : 0;
			selectedBody = 0;
			for (int i = 0; i < kMaxBodies; i++)
			{
				bodyForceX[i] = 0.0f;
				bodyForceY[i] = 0.0f;
				bodyTorque[i] = 0.0f;
				hasForceSample[i] = false;
			}
			if (bodyCount > 0)
			{
				bodies[0] = {
					definition.obstacleStartX,
					definition.obstacleStartY,
					0.0f,
					0.0f,
					definition.obstacleStartX,
					definition.obstacleStartY,
					definition.obstacleRadius,
					true
				};
			}

			// 预留状态复位（功能 4/5）
			dragging = false;
			lButtonDown = rButtonDown = mButtonDown = false;
			activeTool = InteractionTool::None;
			ctrlHeld = false;
			debugCellX = debugCellY = -1;
			// 功能 4：与网格同生命周期复位（§2）
			injectedForceCells.clear();
			prevMouseScreenX = prevMouseScreenY = 0;

			flow = new mrFlow2D();
			flow->Create(
				0.0f,
				0.0f,
				definition.nx,
				definition.ny,
				1.0f,
				(REAL)definition.nx,
				(REAL)definition.ny,
				definition.viscosity,
				0.0f);

			InitializeDemoCase(
				flow, definition, bodies.data(), bodyCount, obstacleShape);

			flows.push_back(flow);
			deviceFlows.push_back(NULL);
			solver.gpuId = 0;
			solver.L = 1.0f;
			solver.AttachLbmHost(flows);
			solver.AttachLbmDevice(deviceFlows);
			solver.mlTransData2Gpu();

			BuildFieldImage(flow, definition, fieldView, pixels);
			initialized = true;
		}

		void SetMoveKey(WPARAM key, bool down)
		{
			if (key == VK_LEFT) moveLeft = down;
			if (key == VK_RIGHT) moveRight = down;
			if (key == VK_UP) moveUp = down;
			if (key == VK_DOWN) moveDown = down;
		}

		void ClearMoveKeys()
		{
			moveLeft = false;
			moveRight = false;
			moveUp = false;
			moveDown = false;
		}

		void UpdateObstacleTarget()
		{
			if (!initialized || bodyCount == 0 || selectedBody < 0 ||
				selectedBody >= bodyCount)
			{
				return;
			}

			float dx = 0.0f;
			float dy = 0.0f;
			if (moveLeft) dx -= 1.0f;
			if (moveRight) dx += 1.0f;
			if (moveDown) dy -= 1.0f;
			if (moveUp) dy += 1.0f;
			if (dx == 0.0f && dy == 0.0f)
			{
				return;
			}

			RigidBody& body = bodies[selectedBody];
			const float length = sqrtf(dx * dx + dy * dy);
			dx = dx / length * definition.obstacleTargetSpeed;
			dy = dy / length * definition.obstacleTargetSpeed;

			const float margin = body.radius + 2.0f;
			body.tx = ClampFloat(
				body.tx + dx,
				margin,
				(float)definition.nx - margin - 1.0f);
			body.ty = ClampFloat(
				body.ty + dy,
				margin,
				(float)definition.ny - margin - 1.0f);

			const float leadX = body.tx - body.x;
			const float leadY = body.ty - body.y;
			const float lead = sqrtf(leadX * leadX + leadY * leadY);
			if (lead > kMaximumTargetLead)
			{
				body.tx = body.x + leadX / lead * kMaximumTargetLead;
				body.ty = body.y + leadY / lead * kMaximumTargetLead;
			}
		}

		void ApplyObstaclePoses(
			const float newX[],
			const float newY[],
			const float wallUx[],
			const float wallUy[])
		{
			const int nx = definition.nx;
			const int ny = definition.ny;
			const int count = nx * ny;
			std::vector<MLLATTICENODE_FLAG> oldFlags(
				flow->flag, flow->flag + count);
			std::vector<MLLATTICENODE_FLAG> newFlags(count);
			std::vector<int> releasedCells;
			std::vector<int> solidCells;
			std::vector<int> touchedCells;
			bool flagsChanged = false;

			// 先把新姿态写入 bodies：重分类与 owner 反查均基于新位置
			for (int i = 0; i < bodyCount; i++)
			{
				bodies[i].x = newX[i];
				bodies[i].y = newY[i];
				bodies[i].vx = wallUx[i];
				bodies[i].vy = wallUy[i];
			}

			// ===== ASSIGNMENT FILL BEGIN: P2-B moving-cylinder coupling =====
			// TODO: Update the lattice representation of the prescribed
			// moving Karman cylinder.
			// 1. Reclassify cells and record newly uncovered fluid cells.
			// 2. Reconstruct uncovered moments from unchanged fluid neighbors.
			// 3. Set solid ghost-cell velocity to the wall velocity while
			//    preserving adjacent non-equilibrium stress.
			// 4. Record all touched cells and update the current obstacle pose.
			// GPU synchronization is performed by the supplied call below.
			{
				// 1. Reclassify every cell and record the flag changes.
				for (int y = 0; y < ny; y++)
				{
					for (int x = 0; x < nx; x++)
					{
						const int idx = y * nx + x;
						const MLLATTICENODE_FLAG baseFlag =
							GetDemoCaseBaseFlag(definition, x, y);
						const bool isSolid = IsAnyObstacleCell(
							bodies.data(), bodyCount, obstacleShape, x, y);
						newFlags[idx] = isSolid ? ML_SOLID : baseFlag;
					}
				}

				for (int idx = 0; idx < count; idx++)
				{
					const bool wasSolid = (oldFlags[idx] == ML_SOLID);
					const bool isSolid = (newFlags[idx] == ML_SOLID);
					if (wasSolid && !isSolid)
					{
						releasedCells.push_back(idx);
					}
					else if (!wasSolid && isSolid)
					{
						solidCells.push_back(idx);
					}
				}

				// 2. Reconstruct uncovered moments from unchanged fluid
				//    neighbors so the released region has valid initial data.
				for (const int idx : releasedCells)
				{
					const int x = idx % nx;
					const int y = idx / nx;
					Moments2D result;
					if (AverageNeighborMoments(
						flow, x, y, oldFlags, newFlags, true, result))
					{
						WriteMoments2D(flow, idx, result.rho, result.ux,
							result.uy, result.sxx, result.syy, result.sxy);
					}
				}

				// Boundary-moment helper mirroring WriteBoundaryMoments:
				// keep the adjacent fluid stress, replace velocity by the wall.
				auto writeBoundaryMoments =
					[&](int boundaryIndex, int neighborIndex,
						REAL targetUx, REAL targetUy)
				{
					REAL rho = flow->fMom[neighborIndex * 6 + 0];
					const REAL neighborUx = flow->fMom[neighborIndex * 6 + 1];
					const REAL neighborUy = flow->fMom[neighborIndex * 6 + 2];
					const REAL neighborSxx = flow->fMom[neighborIndex * 6 + 3];
					const REAL neighborSyy = flow->fMom[neighborIndex * 6 + 4];
					const REAL neighborSxy = flow->fMom[neighborIndex * 6 + 5];

					if (!IsFinite(rho) || rho < 0.5f || rho > 1.5f)
					{
						rho = 1.0f;
					}

					if (!IsFinite(neighborUx) || !IsFinite(neighborUy) ||
						!IsFinite(neighborSxx) || !IsFinite(neighborSyy) ||
						!IsFinite(neighborSxy))
					{
						WriteEquilibriumMoments2D(
							flow, boundaryIndex, rho, targetUx, targetUy);
						return;
					}

					const REAL sxx = targetUx * targetUx +
						(neighborSxx - neighborUx * neighborUx);
					const REAL syy = targetUy * targetUy +
						(neighborSyy - neighborUy * neighborUy);
					const REAL sxy = targetUx * targetUy +
						(neighborSxy - neighborUx * neighborUy);
					WriteMoments2D(flow, boundaryIndex, rho, targetUx,
						targetUy, sxx, syy, sxy);
				};

				// 3. Set each new solid ghost cell to the wall velocity while
				//    preserving the non-equilibrium stress of a fluid neighbor.
				// 反查所属物体取壁面速度；非物体固体格点（owner == -1）壁面速度为 0，
				// 不能假设 owner >= 0（否则 wallUx[-1] 越界）。
				for (const int solidIdx : solidCells)
				{
					const int sx = solidIdx % nx;
					const int sy = solidIdx / nx;
					const int owner = OwnerBodyOfCell(
						bodies.data(), bodyCount, obstacleShape, sx, sy);
					const REAL wallUxCell = (owner >= 0) ? wallUx[owner] : 0.0f;
					const REAL wallUyCell = (owner >= 0) ? wallUy[owner] : 0.0f;
					bool found = false;
					for (int dy = -1; dy <= 1 && !found; dy++)
					{
						for (int dx = -1; dx <= 1; dx++)
						{
							if (dx == 0 && dy == 0) continue;
							const int neighborX = sx + dx;
							const int neighborY = sy + dy;
							if (neighborX < 0 || neighborX >= nx ||
								neighborY < 0 || neighborY >= ny) continue;
							const int neighbor = neighborY * nx + neighborX;
							if (newFlags[neighbor] != ML_FLUID) continue;
							writeBoundaryMoments(
								solidIdx, neighbor, wallUxCell, wallUyCell);
							found = true;
							break;
						}
					}
					if (!found)
					{
						WriteEquilibriumMoments2D(
							flow, solidIdx, 1.0f, wallUxCell, wallUyCell);
					}
				}

				// 4. Record every touched cell and update the obstacle pose.
				touchedCells.insert(touchedCells.end(),
					solidCells.begin(), solidCells.end());
				touchedCells.insert(touchedCells.end(),
					releasedCells.begin(), releasedCells.end());

				for (int idx = 0; idx < count; idx++)
				{
					flow->flag[idx] = newFlags[idx];
				}
				flagsChanged = true;
			}

			// ===== ASSIGNMENT FILL END: P2-B =====

			SyncCellsToDevice(solver, 0, touchedCells, flagsChanged);
		}

		// §2.3 补足：目标点两两互不重叠校验（失败则拒绝本次目标更新）
		// 请求 1：目标点重叠校验。原实现会把两个目标点沿连线互相"推回"，
		// 拖动 A 到 B 上时会连带推动 B 的目标 → B 进入移动状态（bug）。
		// 现改为：目标重叠 → 拒绝本次目标更新（目标复位到当前位置），
		// 不产生任何推动移动；实际运动还受 AdvanceObstacles 候选位置校验拦截。
		void ResolveOverlapTargets()
		{
			for (int i = 0; i < bodyCount; i++)
			{
				for (int j = i + 1; j < bodyCount; j++)
				{
					RigidBody& bi = bodies[i];
					RigidBody& bj = bodies[j];
					if (BodiesOverlap(
						bi.tx, bi.ty, bi.radius,
						bj.tx, bj.ty, bj.radius,
						obstacleShape))
					{
						bi.tx = bi.x;
						bi.ty = bi.y;
						bj.tx = bj.x;
						bj.ty = bj.y;
					}
				}
			}
		}

		void AdvanceObstacles()
		{
			if (!initialized || bodyCount == 0)
			{
				return;
			}

			float newX[kMaxBodies];
			float newY[kMaxBodies];
			float wallUx[kMaxBodies];
			float wallUy[kMaxBodies];
			bool anyMove = false;

			for (int i = 0; i < bodyCount; i++)
			{
				const RigidBody& body = bodies[i];
				const float dx = body.tx - body.x;
				const float dy = body.ty - body.y;
				const float distance = sqrtf(dx * dx + dy * dy);
				if (distance <= 1.0e-5f)
				{
					newX[i] = body.x;
					newY[i] = body.y;
					wallUx[i] = 0.0f;
					wallUy[i] = 0.0f;
					// 位移≈0 但速度非 0 → 仍需调用一次把壁面速度归零
					if (fabsf(body.vx) > 1.0e-6f ||
						fabsf(body.vy) > 1.0e-6f)
					{
						anyMove = true;
					}
					continue;
				}

				const float maxDisplacement = std::min(
					definition.obstacleMoveSpeed,
					kMaximumWallSpeed * (float)stepsPerFrame);
				const float displacement = std::min(distance, maxDisplacement);
				const float moveX = dx / distance * displacement;
				const float moveY = dy / distance * displacement;
				newX[i] = body.x + moveX;
				newY[i] = body.y + moveY;
				wallUx[i] = moveX / (float)stepsPerFrame;
				wallUy[i] = moveY / (float)stepsPerFrame;
				anyMove = true;
			}

			// §2.3-补足：对候选新位置做两两碰撞校验（请求 1：形状感知），
			// 违反的物体放弃本次移动（保留旧位置与目标），不产生推动
			for (int i = 0; i < bodyCount; i++)
			{
				for (int j = i + 1; j < bodyCount; j++)
				{
					if (BodiesOverlap(
						newX[i], newY[i], bodies[i].radius,
						newX[j], newY[j], bodies[j].radius,
						obstacleShape))
					{
						newX[i] = bodies[i].x;
						newY[i] = bodies[i].y;
						wallUx[i] = 0.0f;
						wallUy[i] = 0.0f;
						newX[j] = bodies[j].x;
						newY[j] = bodies[j].y;
						wallUx[j] = 0.0f;
						wallUy[j] = 0.0f;
					}
				}
			}

			ResolveOverlapTargets();

			if (anyMove)
			{
				ApplyObstaclePoses(newX, newY, wallUx, wallUy);
			}
		}

		void ComputeSolidLoads()
		{
			if (bodyCount == 0)
			{
				for (int i = 0; i < kMaxBodies; i++)
				{
					bodyForceX[i] = 0.0f;
					bodyForceY[i] = 0.0f;
					bodyTorque[i] = 0.0f;
					hasForceSample[i] = false;
				}
				return;
			}

			REAL rawForceX[kMaxBodies] = { 0 };
			REAL rawForceY[kMaxBodies] = { 0 };
			REAL rawTorque[kMaxBodies] = { 0 };
			const int nx = definition.nx;
			const int ny = definition.ny;

			for (int y = 1; y < ny - 1; y++)
			{
				for (int x = 1; x < nx - 1; x++)
				{
					const int index = y * nx + x;
					if (flow->flag[index] != ML_FLUID)
					{
						continue;
					}

					const Moments2D fluidMoments = ReadMoments(flow, index);
					if (!AreMomentsUsable(fluidMoments))
					{
						continue;
					}

					for (int direction = 1; direction < 9; direction++)
					{
						const int sourceX = x - (int)ex2d_cpu[direction];
						const int sourceY = y - (int)ey2d_cpu[direction];
						const int source = sourceY * nx + sourceX;
						if (flow->flag[source] != ML_SOLID)
						{
							continue;
						}

						// 反查所属物体；非物体固体格点（owner == -1）不算入任何物体受力
						const int owner = OwnerBodyOfCell(
							bodies.data(), bodyCount, obstacleShape,
							sourceX, sourceY);
						if (owner < 0 || owner >= kMaxBodies)
						{
							continue;
						}

						const Moments2D solidMoments = ReadMoments(flow, source);
						const int inverse = kInverseDirection[direction];
						const REAL outgoing =
							ReconstructDistribution(fluidMoments, inverse);
						const REAL incoming =
							ReconstructDistribution(solidMoments, direction);
						const REAL forceX =
							outgoing * (ex2d_cpu[inverse] - solidMoments.ux) -
							incoming * (ex2d_cpu[direction] - solidMoments.ux);
						const REAL forceY =
							outgoing * (ey2d_cpu[inverse] - solidMoments.uy) -
							incoming * (ey2d_cpu[direction] - solidMoments.uy);
						const REAL boundaryX =
							(REAL)x - 0.5f * ex2d_cpu[direction];
						const REAL boundaryY =
							(REAL)y - 0.5f * ey2d_cpu[direction];

						rawForceX[owner] += forceX;
						rawForceY[owner] += forceY;
						// 力矩参考点为物体自身中心
						rawTorque[owner] +=
							(boundaryX - bodies[owner].x) * forceY -
							(boundaryY - bodies[owner].y) * forceX;
					}
				}
			}

			// 每物体独立平滑滤波；新添加/重置的物体首帧 blend = 1
			for (int i = 0; i < bodyCount; i++)
			{
				const float blend = hasForceSample[i] ? 0.15f : 1.0f;
				bodyForceX[i] += blend * ((float)rawForceX[i] - bodyForceX[i]);
				bodyForceY[i] += blend * ((float)rawForceY[i] - bodyForceY[i]);
				bodyTorque[i] += blend * ((float)rawTorque[i] - bodyTorque[i]);
				hasForceSample[i] = true;
			}
		}

		// §5.2 工具统一入口：cx/cy = 格点坐标（mouseFieldX/Y），pressed = 是否按下。
		// 仅 Blow/Vortex 写 forcex/forcey 并收集到 injectedForceCells 供
		// ClearInjectedForces 本帧末清零。
		// 注意：不在开头 clear injectedForceCells——同一帧内左键工具 + 中键旋涡
		// 可能同时执行，第二个工具若 clear 会丢失第一个的格点 → 漏清力（§10.11）。
		void ApplyToolAt(float cx, float cy, InteractionTool tool, bool pressed)
		{
			const int nx = definition.nx;
			const int ny = definition.ny;
			const int radius = (int)std::ceil(brushRadius);
			const float radius2 = brushRadius * brushRadius;

			for (int y = (int)cy - radius; y <= (int)cy + radius; y++)
			{
				for (int x = (int)cx - radius; x <= (int)cx + radius; x++)
				{
					if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1) continue;
					const int idx = y * nx + x;
					const float dx = (float)x - cx;
					const float dy = (float)y - cy;
					if (dx * dx + dy * dy > radius2) continue;
					// 边界格点（INLET/OUTLET/WALL_*）禁止施力（沿用"边界禁画"原则）
					if (flow->flag[idx] != ML_FLUID) continue;

					switch (tool)
					{
					case InteractionTool::Blow:
					{
						// §5.5 定向体力：方向 = 帧间鼠标差分（blowDirX/Y），强度 blowStrength
						const float dist = sqrtf(dx * dx + dy * dy);
						const float falloff = std::max(0.0f,
							1.0f - dist / brushRadius);
						flow->forcex[idx] = ClampFloat(
							blowDirX * blowStrength * falloff, -kFmax, kFmax);
						flow->forcey[idx] = ClampFloat(
							blowDirY * blowStrength * falloff, -kFmax, kFmax);
						injectedForceCells.push_back(idx);
						break;
					}
					case InteractionTool::Vortex:
					{
						// §5.5 切向体力场（逆时针）；k 与 blowStrength 同量级
						const float dist = std::max(1.0e-4f, sqrtf(dx * dx + dy * dy));
						const float falloff = std::max(0.0f,
							1.0f - dist / brushRadius);
						flow->forcex[idx] = ClampFloat(
							-blowStrength * (dy / dist) * falloff, -kFmax, kFmax);
						flow->forcey[idx] = ClampFloat(
							blowStrength * (dx / dist) * falloff, -kFmax, kFmax);
						injectedForceCells.push_back(idx);
						break;
					}
					case InteractionTool::None:
					default:
						break;
					}
				}
			}

			if (!injectedForceCells.empty())
			{
				SyncCellsToDevice(solver, 0, injectedForceCells, /*syncFlags=*/false);
			}
		}

		// §5.3 清力：与 WriteMoments2D（只清边界/固体格点）互补——本函数只清
		// brush 写入的流体格点，保证注入的力"只活一帧"。
		void ClearInjectedForces()
		{
			if (injectedForceCells.empty())
			{
				return;
			}
			for (const int idx : injectedForceCells)
			{
				flow->forcex[idx] = 0.0f;
				flow->forcey[idx] = 0.0f;
			}
			SyncCellsToDevice(solver, 0, injectedForceCells, /*syncFlags=*/false);
			injectedForceCells.clear();
		}

		// 功能 4（§5.1/§5.4/§5.5）：交互状态机。每 UI 帧由 Step() 调用一次，
		// 位于 LBM 循环之前：本帧注入的力本帧就被流场响应（视觉跟手）。
		void ApplyMouseEffects()
		{
			if (flow == NULL)
			{
				return;
			}

			// §5.5 差分更新：无条件执行，保证差分是"帧间增量"而非"按下以来总位移"
			const int dScreenX = mouseScreenX - prevMouseScreenX;
			const int dScreenY = mouseScreenY - prevMouseScreenY;
			prevMouseScreenX = mouseScreenX;
			prevMouseScreenY = mouseScreenY;
			// 屏幕 y 向下、格点 y 向上：对屏幕 dy 取反后才是格点方向（视觉一致）
			blowDirX = 0.0f;
			blowDirY = 1.0f;   // 未动 → 主流方向（Karman/Jet 均 +y 上行）
			if (dScreenX != 0 || dScreenY != 0)
			{
				const float len = sqrtf(
					(float)(dScreenX * dScreenX + dScreenY * dScreenY));
				blowDirX = (float)dScreenX / len;
				blowDirY = -(float)dScreenY / len;
			}

			// §5.4 面板豁免（双保险）：所有工具执行前必须现算 layout
			const UiLayout layout = CurrentLayout(appWindow);
			const bool overPanel =
				IsOverUiPanel(mouseScreenX, mouseScreenY, layout);
			const bool uiCaptures = UiWantsCaptureMouse();

			// 左键：Shift+左键 = Blow（覆盖 activeTool）；未命中物体时按 activeTool
			if (lButtonDown && !dragging && !overPanel && !uiCaptures)
			{
				const bool shiftHeld = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
				InteractionTool tool =
					shiftHeld ? InteractionTool::Blow : activeTool;
				if (tool != InteractionTool::None)
				{
					ApplyToolAt(mouseFieldX, mouseFieldY, tool, true);
				}
			}

			// 中键：旋涡（不依赖工具选择，恒可用）
			if (mButtonDown && !overPanel && !uiCaptures)
			{
				ApplyToolAt(
					mouseFieldX, mouseFieldY, InteractionTool::Vortex, true);
			}
		}

		void Step()
		{
			if (!initialized)
			{
				return;
			}

			// 功能 4/6 挂点：LBM 循环之前的鼠标效果（注入烟 + 写力）
			ApplyMouseEffects();

			RefreshDemoCaseBoundaries(
				flow, definition, iteration);
			SyncOuterBoundaryMomentsToDevice(solver, 0);

			for (int step = 0; step < stepsPerFrame; step++)
			{
				solver.mlIterateGpu();
				iteration++;
			}

			CopyMomentsFromDevice(solver, 0);
			ClearInjectedForces();   // ③ 清本帧注入的力 + 同步（§5.3，力"只活一帧"）
			ComputeSolidLoads();
			BuildFieldImage(flow, definition, fieldView, pixels);
			frame++;
		}

		void ToggleFieldView()
		{
			// 功能 4：三视图循环 Velocity → Vorticity → Colorful → Velocity（V 键）
			fieldView = fieldView == DemoFieldView::VelocityMagnitude
				? DemoFieldView::Vorticity
				: (fieldView == DemoFieldView::Vorticity
					? DemoFieldView::Colorful
					: DemoFieldView::VelocityMagnitude);
			if (initialized)
			{
				BuildFieldImage(flow, definition, fieldView, pixels);
			}
		}

		// §3.2 命中检测：从后往前遍历（后添加者优先），容差 radius + 3 格
		int PickBody(float gx, float gy)
		{
			for (int i = bodyCount - 1; i >= 0; i--)
			{
				const RigidBody& b = bodies[i];
				const float dx = gx - b.x;
				const float dy = gy - b.y;
				const float reach = b.radius + 3.0f;
				if (dx * dx + dy * dy <= reach * reach)
				{
					return i;
				}
			}
			return -1;
		}

		// §2.2 形状切换 / 增删物体后的全量重分类：以当前姿态调用一次
		void SyncBodiesToFlow()
		{
			if (!initialized)
			{
				return;
			}
			float newX[kMaxBodies] = {};
			float newY[kMaxBodies] = {};
			float wallUx[kMaxBodies] = {};
			float wallUy[kMaxBodies] = {};
			for (int i = 0; i < bodyCount; i++)
			{
				newX[i] = bodies[i].x;
				newY[i] = bodies[i].y;
			}
			ApplyObstaclePoses(newX, newY, wallUx, wallUy);
		}

		// §2.5 候选位置校验：与现有物体当前位置（而非目标 tx/ty）距离校验
		bool CanPlaceBody(float x, float y, float radius) const
		{
			if (bodyCount >= kMaxBodies)
			{
				return false;
			}
			if (!(radius > 0.0f) || 2.0f * radius + 2.0f >
				(float)std::min(definition.nx, definition.ny))
			{
				return false;
			}
			const float margin = radius + 2.0f;
			if (x < margin || x > (float)definition.nx - margin - 1.0f ||
				y < margin || y > (float)definition.ny - margin - 1.0f)
			{
				return false;
			}
			for (int j = 0; j < bodyCount; j++)
			{
				const RigidBody& b = bodies[j];
				// 请求 1：形状感知碰撞（+1 安全间距，避免贴着放置后格点接壤）
				if (BodiesOverlap(
					x, y, radius + 1.0f,
					b.x, b.y, b.radius + 1.0f,
					obstacleShape))
				{
					return false;
				}
			}
			return true;
		}

		// §2.5 添加物体：重叠校验失败返回 false（UI 提示且不添加）
		bool AddBodyAt(float x, float y, float radius)
		{
			if (!CanPlaceBody(x, y, radius))
			{
				return false;
			}
			const int index = bodyCount;
			bodies[index] = {
				x, y, 0.0f, 0.0f, x, y, radius, false
			};
			bodyCount++;
			bodyForceX[index] = bodyForceY[index] = bodyTorque[index] = 0.0f;
			hasForceSample[index] = false;
			SelectBody(index);
			SyncBodiesToFlow();
			return true;
		}

		// §2.5 删除物体：换位删除 + 维护选中态（bodyCount == 0 时 selectedBody = -1）
		void RemoveBody(int index)
		{
			if (bodyCount == 0 || index < 0 || index >= bodyCount)
			{
				return;
			}
			bodies[index] = bodies[bodyCount - 1];
			bodyCount--;
			if (bodyCount == 0)
			{
				selectedBody = -1;
				SyncBodiesToFlow();   // 释放被删物体的全部固体格点
				return;
			}
			if (selectedBody >= bodyCount)
			{
				selectedBody = bodyCount - 1;
			}
			SelectBody(selectedBody);
			SyncBodiesToFlow();
		}

		// 重建 selected 标志（仅 selectedBody 为 true）
		void SelectBody(int index)
		{
			selectedBody = index;
			for (int i = 0; i < bodyCount; i++)
			{
				bodies[i].selected = (i == index);
			}
		}

		// §2.5 Tab 循环切换选中物体
		void CycleSelection()
		{
			if (bodyCount == 0)
			{
				return;
			}
			SelectBody((selectedBody + 1) % bodyCount);
		}

		// §4.1 数据同源：UI 改 bodies[0] 半径/中心时同步 definition（Re 显示不失真）
		void SyncBodyToDefinition()
		{
			if (bodyCount > 0)
			{
				definition.obstacleRadius = bodies[0].radius;
				definition.obstacleStartX = bodies[0].x;
				definition.obstacleStartY = bodies[0].y;
			}
		}
	};

	LbmApp gApp;
	DemoCaseId gStartupCase = DemoCaseId::KarmanVortex;
	bool gHelpRequested = false;

	// §4.3 启动加载：main() 在 ParseArguments 之后读 params.ini 存入这里，
	// WM_CREATE → gApp.Init + InitOpenGL 之后按 §4.1 三分类分发应用。
	struct LoadedStartupParams
	{
		bool valid = false;
		DemoCaseId caseId = DemoCaseId::KarmanVortex;
		DemoCaseDefinition def;
		DemoFieldView view = DemoFieldView::Vorticity;
		int stepsPerFrame = 5;
		ObstacleShape shape = ObstacleShape::Circle;
		int bodyCount = 0;
		std::array<RigidBody, kMaxBodies> bodies = {};
	};
	LoadedStartupParams gLoadedParams;

	// §4.3 优先级约定：命令行显式 --case 时 ini 不覆盖 case（只覆盖其余参数）
	bool gCaseSpecifiedOnCommandLine = false;

	// §4.2 当前预设索引（Karman / Jet / Custom）
	int gActivePreset = 0;

	// 请求 1：UI / 窗口大小模式（小=当前默认窗口 1040×820，中=中等窗口，
	// 大=全屏）。通过右侧面板的 UI size 栏切换。
	enum class UiSizeMode { Small, Medium, Large };
	UiSizeMode gUiSizeMode = UiSizeMode::Small;

	// 请求 3：UI 尺寸缩放系数（文字 + 控件/按钮尺寸随模式等比放大）
	float UiScale()
	{
		switch (gUiSizeMode)
		{
		case UiSizeMode::Large:  return 1.35f;
		case UiSizeMode::Medium: return 1.15f;
		case UiSizeMode::Small:
		default:                 return 1.0f;
		}
	}
	HDC gDeviceContext = NULL;
	HGLRC gOpenGlContext = NULL;
	GLuint gFieldTexture = 0;
	GLuint gLegendTexture = 0;   // §5.5 色条纹理（256×1，随 fieldView 重建）
	GLuint gFontBase = 0;

	std::wstring BuildWindowTitle()
	{
		std::wstringstream title;
		title << L"Home2D LBM | "
			<< (gApp.caseId == DemoCaseId::KarmanVortex
				? L"Karman vortex"
				: L"Jet flow")
			<< L" | "
			<< (gApp.paused ? L"Paused" : L"Running")
			<< L" | iter=" << gApp.iteration
			<< L" | steps/frame=" << gApp.stepsPerFrame;
		return title.str();
	}

	void UpdateWindowTitle(HWND window)
	{
		SetWindowTextW(window, BuildWindowTitle().c_str());
	}

	bool CreateFontLists()
	{
		HFONT font = CreateFontA(
			16,
			0,
			0,
			0,
			FW_NORMAL,
			FALSE,
			FALSE,
			FALSE,
			ANSI_CHARSET,
			OUT_DEFAULT_PRECIS,
			CLIP_DEFAULT_PRECIS,
			CLEARTYPE_QUALITY,
			FIXED_PITCH | FF_MODERN,
			"Consolas");
		if (font == NULL)
		{
			return false;
		}

		HGDIOBJ oldFont = SelectObject(gDeviceContext, font);
		gFontBase = glGenLists(96);
		const BOOL created =
			wglUseFontBitmapsA(gDeviceContext, 32, 96, gFontBase);
		SelectObject(gDeviceContext, oldFont);
		DeleteObject(font);
		return created == TRUE;
	}

	void CreateFieldTexture()
	{
		if (gFieldTexture == 0)
		{
			glGenTextures(1, &gFieldTexture);
		}

		glBindTexture(GL_TEXTURE_2D, gFieldTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			gApp.definition.nx,
			gApp.definition.ny,
			0,
			GL_RGB,
			GL_UNSIGNED_BYTE,
			gApp.pixels.data());
	}

	// §5.5 色条纹理重建：内容随 fieldView 变化（Velocity→MAGMA / Vorticity→MapVorticityColor）
	void RebuildLegendTexture()
	{
		if (gLegendTexture == 0)
		{
			glGenTextures(1, &gLegendTexture);
		}

		std::vector<unsigned char> pixels(256 * 3);
		ColorRamp colorRamp;
		for (int i = 0; i < 256; i++)
		{
			float r = 0.0f;
			float g = 0.0f;
			float b = 0.0f;
			if (gApp.fieldView == DemoFieldView::VelocityMagnitude)
			{
				const float normalized = (float)i / 255.0f;
				vec3 color(0.0f, 0.0f, 0.0f);
				colorRamp.set_GLcolor(normalized, COLOR__MAGMA, color, false);
				r = color.x;
				g = color.y;
				b = color.z;
			}
			else if (gApp.fieldView == DemoFieldView::Colorful)
			{
				// 功能 4 第 3 点：色条显示"时间对应的默认颜色"——随时间缓慢旋转的色相。
				// 旋转速率与 MapFlowFieldColor 一致（0.0167 rad/s，约 60s 一圈），
				// 色相基址同样含 +0.5 偏移；白色端 = 相对速度 0（静止/稳定同向），
				// 右端 = 当前时刻的全鲜艳色。Render 每帧重建以跟随时间。
				const float timeSeconds =
					(float)(GetTickCount() & 0x3FFFFFFF) * 0.001f;
				const float t = (float)i / 255.0f;
				float hue = 0.5f + 0.0167f * timeSeconds;
				hue -= floorf(hue);
				float vr = 1.0f, vg = 1.0f, vb = 1.0f;
				HsvToRgb(hue, 1.0f, 1.0f, vr, vg, vb);
				r = 1.0f * (1.0f - t) + vr * t;
				g = 1.0f * (1.0f - t) + vg * t;
				b = 1.0f * (1.0f - t) + vb * t;
			}
			else
			{
				const float normalized = -1.0f + 2.0f * (float)i / 255.0f;
				MapVorticityColor(normalized, r, g, b);
			}
			pixels[i * 3 + 0] = (unsigned char)(ClampFloat(r, 0.0f, 1.0f) * 255.0f);
			pixels[i * 3 + 1] = (unsigned char)(ClampFloat(g, 0.0f, 1.0f) * 255.0f);
			pixels[i * 3 + 2] = (unsigned char)(ClampFloat(b, 0.0f, 1.0f) * 255.0f);
		}

		glBindTexture(GL_TEXTURE_2D, gLegendTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			256,
			1,
			0,
			GL_RGB,
			GL_UNSIGNED_BYTE,
			pixels.data());
	}

	bool InitOpenGL(HWND window)
	{
		gDeviceContext = GetDC(window);
		PIXELFORMATDESCRIPTOR format = {};
		format.nSize = sizeof(format);
		format.nVersion = 1;
		format.dwFlags =
			PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
		format.iPixelType = PFD_TYPE_RGBA;
		format.cColorBits = 24;
		format.iLayerType = PFD_MAIN_PLANE;

		const int pixelFormat =
			ChoosePixelFormat(gDeviceContext, &format);
		if (pixelFormat == 0 ||
			!SetPixelFormat(gDeviceContext, pixelFormat, &format))
		{
			return false;
		}

		gOpenGlContext = wglCreateContext(gDeviceContext);
		if (gOpenGlContext == NULL ||
			!wglMakeCurrent(gDeviceContext, gOpenGlContext))
		{
			return false;
		}

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		CreateFieldTexture();
		RebuildLegendTexture();

		// §5.1 集成：ImGui 上下文 + Win32/OpenGL2 后端
		//（禁用 imgui.ini，布局由代码固定，避免写工作目录）
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.IniFilename = NULL;
		ImGui_ImplWin32_Init(window);
		ImGui_ImplOpenGL2_Init();

		return CreateFontLists();
	}

	void ShutdownOpenGL(HWND window)
	{
		// §5.1 反初始化：顺序与 Init 相反
		ImGui_ImplOpenGL2_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();

		if (gFieldTexture != 0)
		{
			glDeleteTextures(1, &gFieldTexture);
			gFieldTexture = 0;
		}
		if (gLegendTexture != 0)
		{
			glDeleteTextures(1, &gLegendTexture);
			gLegendTexture = 0;
		}
		if (gFontBase != 0)
		{
			glDeleteLists(gFontBase, 96);
			gFontBase = 0;
		}
		if (gOpenGlContext != NULL)
		{
			wglMakeCurrent(NULL, NULL);
			wglDeleteContext(gOpenGlContext);
			gOpenGlContext = NULL;
		}
		if (gDeviceContext != NULL)
		{
			ReleaseDC(window, gDeviceContext);
			gDeviceContext = NULL;
		}
	}

	void BeginScreenCoordinates(int width, int height)
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	void DrawSolidRect(const RectF& rect, float r, float g, float b, float a)
	{
		glColor4f(r, g, b, a);
		glBegin(GL_QUADS);
		glVertex2f(rect.left, rect.top);
		glVertex2f(rect.right, rect.top);
		glVertex2f(rect.right, rect.bottom);
		glVertex2f(rect.left, rect.bottom);
		glEnd();
	}

	void DrawLine(
		float x0,
		float y0,
		float x1,
		float y1,
		float r,
		float g,
		float b)
	{
		glColor3f(r, g, b);
		glBegin(GL_LINES);
		glVertex2f(x0, y0);
		glVertex2f(x1, y1);
		glEnd();
	}

	void DrawText(float x, float y, const std::string& text)
	{
		if (gFontBase == 0 || text.empty())
		{
			return;
		}
		glRasterPos2f(x, y);
		glPushAttrib(GL_LIST_BIT);
		glListBase(gFontBase - 32);
		glCallLists((GLsizei)text.size(), GL_UNSIGNED_BYTE, text.c_str());
		glPopAttrib();
	}

	UiLayout ComputeLayout(int width, int height)
	{
		const float margin = 22.0f;
		const float gap = 22.0f;
		// 请求 3：面板宽度随 UI 尺寸模式缩放（给放大的文字/按钮留空间）
		const float panelWidth = (float)std::min(
			(int)(kPanelWidth * UiScale()), std::max(300, width / 3));
		UiLayout layout;
		layout.panel.left = (float)width - panelWidth;
		layout.panel.top = 0.0f;
		layout.panel.right = (float)width;
		layout.panel.bottom = (float)height;

		const float areaLeft = margin;
		const float areaTop = margin;
		const float areaRight = layout.panel.left - gap;
		const float areaBottom = (float)height - margin;
		const float availableWidth = std::max(1.0f, areaRight - areaLeft);
		const float availableHeight = std::max(1.0f, areaBottom - areaTop);
		const float fieldAspect =
			(float)gApp.definition.nx / (float)gApp.definition.ny;

		float fieldWidth = availableHeight * fieldAspect;
		float fieldHeight = availableHeight;
		if (fieldWidth > availableWidth)
		{
			fieldWidth = availableWidth;
			fieldHeight = fieldWidth / fieldAspect;
		}

		const float centerX = 0.5f * (areaLeft + areaRight);
		const float centerY = 0.5f * (areaTop + areaBottom);
		layout.field.left = centerX - 0.5f * fieldWidth;
		layout.field.right = centerX + 0.5f * fieldWidth;
		layout.field.top = centerY - 0.5f * fieldHeight;
		layout.field.bottom = centerY + 0.5f * fieldHeight;
		return layout;
	}

	void DrawField(const RectF& field)
	{
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, gFieldTexture);
		glTexSubImage2D(
			GL_TEXTURE_2D,
			0,
			0,
			0,
			gApp.definition.nx,
			gApp.definition.ny,
			GL_RGB,
			GL_UNSIGNED_BYTE,
			gApp.pixels.data());
		glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(field.left, field.top);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(field.right, field.top);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(field.right, field.bottom);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(field.left, field.bottom);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		glColor3f(0.35f, 0.39f, 0.43f);
		glBegin(GL_LINE_LOOP);
		glVertex2f(field.left, field.top);
		glVertex2f(field.right, field.top);
		glVertex2f(field.right, field.bottom);
		glVertex2f(field.left, field.bottom);
		glEnd();
	}

	void DrawBodies(const RectF& field)
	{
		if (gApp.bodyCount == 0)
		{
			return;
		}

		const float scaleX =
			field.Width() / (float)(gApp.definition.nx - 1);
		const float scaleY =
			field.Height() / (float)(gApp.definition.ny - 1);

		for (int i = 0; i < gApp.bodyCount; i++)
		{
			const RigidBody& body = gApp.bodies[i];
			const float x = field.left + body.x * scaleX;
			const float y = field.bottom - body.y * scaleY;
			const float radius = body.radius * scaleX;

			if (body.selected)
			{
				glColor3f(0.23f, 0.80f, 0.82f);
			}
			else
			{
				glColor3f(0.55f, 0.58f, 0.62f);
			}

			glBegin(GL_LINE_LOOP);
			switch (gApp.obstacleShape)
			{
			case ObstacleShape::Circle:
				for (int j = 0; j < 48; j++)
				{
					const float angle =
						6.28318530717958647692f * (float)j / 48.0f;
					glVertex2f(
						x + cosf(angle) * radius,
						y + sinf(angle) * radius);
				}
				break;
			case ObstacleShape::Box:
				glVertex2f(x - radius, y - radius);
				glVertex2f(x + radius, y - radius);
				glVertex2f(x + radius, y + radius);
				glVertex2f(x - radius, y + radius);
				break;
			case ObstacleShape::Diamond:
				glVertex2f(x, y - radius);
				glVertex2f(x + radius, y);
				glVertex2f(x, y + radius);
				glVertex2f(x - radius, y);
				break;
			}
			glEnd();

			if (body.selected)
			{
				DrawLine(x - 5.0f, y, x + 5.0f, y, 0.23f, 0.80f, 0.82f);
				DrawLine(x, y - 5.0f, x, y + 5.0f, 0.23f, 0.80f, 0.82f);
			}
		}
	}

	// §1.3 屏幕像素 → 格点坐标（float），与 DrawBodies 的换算互逆
	void ScreenToFieldPoint(
		int mx,
		int my,
		const UiLayout& layout,
		float& gx,
		float& gy)
	{
		gx = (mx - layout.field.left) / layout.field.Width() *
			((float)gApp.definition.nx - 1.0f);
		gy = (layout.field.bottom - my) / layout.field.Height() *
			((float)gApp.definition.ny - 1.0f);
	}

	// §5.4 面板豁免（双保险之一）：坐标落在右侧 UI 面板矩形内
	bool IsOverUiPanel(int mx, int my, const UiLayout& layout)
	{
		return mx >= layout.panel.left && mx <= layout.panel.right &&
			my >= layout.panel.top && my <= layout.panel.bottom;
	}

	// §3.1-补足 2 / §5.4：ImGui 捕获鼠标判定（面板豁免第一重保险，
	// 与 IsOverUiPanel 的 panel 矩形双保险配合）
	bool UiWantsCaptureMouse()
	{
		return ImGui::GetIO().WantCaptureMouse;
	}

	// §5.4：ImGui 捕获键盘判定（数值输入框等场景快捷键让位）
	bool UiWantsCaptureKeyboard()
	{
		return ImGui::GetIO().WantCaptureKeyboard;
	}

	// 鼠标消息内现算 layout（窗口可能已缩放，不能缓存旧 layout，§3.1-补足 3）
	UiLayout CurrentLayout(HWND window)
	{
		RECT client;
		GetClientRect(window, &client);
		return ComputeLayout(
			client.right - client.left,
			client.bottom - client.top);
	}

	// 功能 5 预留挂点（§1.4）：本期为空实现
	void DrawDebugOverlay(const UiLayout& layout)
	{
		// TODO(M6): Ctrl + 点击格点调试信息 / 放大镜
	}

	// 请求 4：色条浮层——固定在画面（流场）右下角，随 fieldView 显示对应色标。
	// 不再作为 ImGui 面板组件放在中间区域。
	// 请求 2：去掉文字提示与底衬，只保留色条本身。
	void DrawLegendOverlay(const RectF& field)
	{
		if (gLegendTexture == 0)
		{
			return;
		}

		const float barWidth = 240.0f;
		const float barHeight = 14.0f;
		const float margin = 12.0f;
		const float x0 = field.right - margin - barWidth;
		const float y0 = field.bottom - margin - barHeight;

		// 色条本体（256×1 纹理横向拉伸）
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, gLegendTexture);
		glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(x0, y0);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(x0 + barWidth, y0);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(x0 + barWidth, y0 + barHeight);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(x0, y0 + barHeight);
		glEnd();
		glDisable(GL_TEXTURE_2D);
	}

	// 前置声明（BuildAppUi 定义在 ResetSimulation / AdvanceOneUiFrame 之前）
	void ResetSimulation(HWND window, DemoCaseId caseId);
	void AdvanceOneUiFrame();
	// §4.2/§4.3 预设应用与参数分发（定义在 ResetSimulation 之后，需在 BuildAppUi 中调用）
	void ApplyPreset(HWND window, int presetIndex);
	// 请求 2：forceRebuild=true 时无条件重建算例（Restart：以当前设置重新开始）
	void ApplyLoadedStartupParams(HWND window, bool forceRebuild = false);
	// 请求 1：UI 大小模式切换（小/中/大，大=全屏）
	void ApplyUiSizeMode(HWND window, UiSizeMode mode);
	// 请求 2/4：以当前全部设置重建算例（Jet 调参 / Restart 共用）
	void RestartWithCurrentSettings(HWND window);

	// §5.2/§5.3 右侧 ImGui 面板：UI SIZE / SIMULATION / CASE·PRESET / PARAMETERS
	// / OBJECTS / TOOLS(预留) / EDIT MODE(预留) / DEBUG(预留) / CONTROLS。
	// 色条（§5.5）已移出面板，由 DrawLegendOverlay 固定在画面右下角（请求 4）。
	void BuildAppUi(HWND window, const UiLayout& layout)
	{
		ImGui::SetNextWindowPos(ImVec2(layout.panel.left, 0.0f));
		ImGui::SetNextWindowSize(
			ImVec2(layout.panel.Width(), layout.panel.Height()));
		ImGui::Begin(
			"Home2D Controls",
			NULL,
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoTitleBar);

		// ---- UI SIZE（请求 1：小/中/大，大=全屏）----
		ImGui::Text("UI size");
		int uiSizeIndex = (int)gUiSizeMode;
		ImGui::RadioButton("Small", &uiSizeIndex, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Medium", &uiSizeIndex, 1);
		ImGui::SameLine();
		ImGui::RadioButton("Large", &uiSizeIndex, 2);
		if (uiSizeIndex != (int)gUiSizeMode)
		{
			ApplyUiSizeMode(window, (UiSizeMode)uiSizeIndex);
		}
		ImGui::Separator();

		// ---- SIMULATION ----
		if (ImGui::CollapsingHeader("SIMULATION", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::Text("%s  |  iter=%d  frame=%d",
				gApp.paused ? "Paused" : "Running",
				gApp.iteration, gApp.frame);
			ImGui::SliderInt(
				"steps/frame",
				&gApp.stepsPerFrame,
				kMinimumStepsPerFrame,
				kMaximumStepsPerFrame);
			if (ImGui::Button(gApp.paused ? "Resume" : "Pause"))
			{
				gApp.paused = !gApp.paused;
				UpdateWindowTitle(window);
			}
			ImGui::SameLine();
			if (ImGui::Button("Step"))
			{
				AdvanceOneUiFrame();
				UpdateWindowTitle(window);
			}
			ImGui::SameLine();
			if (ImGui::Button("Reset"))
			{
				ResetSimulation(window, gApp.caseId);
			}
			// 请求 2：以当前设置重建算例（不改动已调参数），
			// 适配 Jet flow 调整不同参数后反复观察初始流场状态
			ImGui::SameLine();
			if (ImGui::Button("Restart"))
			{
				RestartWithCurrentSettings(window);
			}

			const char* viewItems[] = { "Velocity", "Vorticity", "Colorful" };
			int viewIndex = (int)gApp.fieldView;
			if (ImGui::Combo("Field view", &viewIndex, viewItems, 3))
			{
				if (viewIndex != (int)gApp.fieldView)
				{
					gApp.fieldView = (DemoFieldView)viewIndex;
					if (gApp.initialized)
					{
						BuildFieldImage(
							gApp.flow, gApp.definition, gApp.fieldView, gApp.pixels);
					}
					RebuildLegendTexture();   // §5.5 色条内容随视图切换
				}
			}
		}

		// ---- CASE / PRESET ----
		if (ImGui::CollapsingHeader("CASE / PRESET", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const char* caseItems[] = { "Karman vortex", "Jet flow" };
			int caseIndex = gApp.caseId == DemoCaseId::KarmanVortex ? 0 : 1;
			if (ImGui::Combo("Case", &caseIndex, caseItems, 2))
			{
				ResetSimulation(
					window,
					caseIndex == 0
					? DemoCaseId::KarmanVortex
					: DemoCaseId::JetFlow);
				gActivePreset = 2;   // 已偏离内置预设 → 标记 Custom
			}

			// §4.2 预设下拉（选择即应用，等效 §5.3 的 [Apply]）
			const char* presetItems[] =
			{
				"Karman", "Jet", "Custom"
			};
			int presetIndex = gActivePreset;
			if (ImGui::Combo("Preset", &presetIndex, presetItems, 3))
			{
				if (presetIndex != gActivePreset)
				{
					ApplyPreset(window, presetIndex);
				}
			}

			ImGui::Text("Reynolds number: %.1f",
				GetDemoCaseReynoldsNumber(gApp.definition));
			if (ImGui::Button("Restore defaults"))
			{
				ResetSimulation(window, gApp.caseId);
				// 恢复默认 = 回到该 case 的基础预设（Karman=0 / Jet=1）
				gActivePreset =
					gApp.caseId == DemoCaseId::KarmanVortex ? 0 : 1;
			}
			ImGui::SameLine();
			// §4.3 保存/读取参数文件（ParamsIO，M4c）
			if (ImGui::Button("Save ini"))
			{
				if (!SaveAppParams(
					"params.ini",
					gApp.definition,
					gApp.fieldView,
					gApp.stepsPerFrame,
					gApp.obstacleShape,
					gApp.bodyCount,
					gApp.bodies.data()))
				{
					ImGui::OpenPopup("params_save_failed");
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("Load ini"))
			{
				// 以当前 case 为基底读取，面板内允许 ini 的 case 生效
				LoadedStartupParams loaded;
				loaded.caseId = gApp.caseId;
				loaded.valid = LoadAppParams(
					"params.ini",
					loaded.caseId,
					loaded.def,
					loaded.view,
					loaded.stepsPerFrame,
					loaded.shape,
					loaded.bodyCount,
					loaded.bodies.data(),
					true);
				if (loaded.valid)
				{
					gLoadedParams = loaded;
					ApplyLoadedStartupParams(window);
					UpdateWindowTitle(window);
					// 与默认 def 一致时归位对应基础预设，否则视为 Custom
					gActivePreset = 2;
					if (loaded.caseId == DemoCaseId::KarmanVortex &&
						loaded.def.viscosity ==
							GetDefaultDefinition(DemoCaseId::KarmanVortex).viscosity)
					{
						gActivePreset = 0;
					}
					else if (loaded.caseId == DemoCaseId::JetFlow &&
						loaded.def.viscosity ==
							GetDefaultDefinition(DemoCaseId::JetFlow).viscosity)
					{
						gActivePreset = 1;
					}
				}
				else
				{
					ImGui::OpenPopup("params_load_failed");
				}
			}
			if (ImGui::BeginPopup("params_save_failed"))
			{
				ImGui::Text("Failed to write params.ini");
				ImGui::EndPopup();
			}
			if (ImGui::BeginPopup("params_load_failed"))
			{
				ImGui::Text("params.ini not found or unreadable");
				ImGui::EndPopup();
			}
		}

		// ---- PARAMETERS ----
		if (ImGui::CollapsingHeader("PARAMETERS", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// 粘度：写入 host 后同步设备（§4.1 第二类）
			float viscosity = (float)gApp.definition.viscosity;
			if (ImGui::SliderFloat(
				"Viscosity", &viscosity, 0.001f, 0.05f, "%.4f"))
			{
				gApp.definition.viscosity = viscosity;
				if (gApp.flow != NULL)
				{
					gApp.flow->vis_shear = viscosity;
					SyncFlowScalarsToDevice(gApp.solver, 0);
				}
			}
			// 直接生效参数（§4.1 第一类）
			// 请求 1：inlet ux 允许为负（负值 = 流向左偏），滑块以 0 居中对称。
			float inletUx = (float)gApp.definition.inletUx;
			if (ImGui::SliderFloat(
				"Inlet ux (0 centered, neg = left)", &inletUx, -0.2f, 0.2f, "%.4f"))
			{
				gApp.definition.inletUx = inletUx;
			}
			float inletUy = (float)gApp.definition.inletUy;
			if (ImGui::SliderFloat(
				"Inlet uy", &inletUy, 0.0f, 0.2f, "%.4f"))
			{
				gApp.definition.inletUy = inletUy;
			}
			float perturbation =
				(float)gApp.definition.inletPerturbationAmplitude;
			if (ImGui::SliderFloat(
				"Perturb. amp", &perturbation, 0.0f, 0.01f, "%.5f"))
			{
				gApp.definition.inletPerturbationAmplitude = perturbation;
			}
			int perturbationPeriod = gApp.definition.inletPerturbationPeriod;
			if (ImGui::SliderInt(
				"Perturb. period", &perturbationPeriod, 10, 1000))
			{
				gApp.definition.inletPerturbationPeriod = perturbationPeriod;
			}
			// 请求 3：颜色上限只在对应显示模式下展示
			//（Velocity 视图只显示 Speed color max，Vorticity 视图只显示 Vorticity color max；
			//  Colorful 视图显示颜色鲜艳度滑块——越低越白、越高越鲜艳）
			if (gApp.fieldView == DemoFieldView::VelocityMagnitude)
			{
				float speedMax = gApp.definition.speedColorMax;
				if (ImGui::SliderFloat(
					"Speed color max", &speedMax, 0.01f, 0.5f, "%.3f"))
				{
					gApp.definition.speedColorMax = speedMax;
				}
			}
			else if (gApp.fieldView == DemoFieldView::Vorticity)
			{
				float vorticityMax = gApp.definition.vorticityColorMax;
				if (ImGui::SliderFloat(
					"Vorticity color max", &vorticityMax, 0.01f, 0.5f, "%.3f"))
				{
					gApp.definition.vorticityColorMax = vorticityMax;
				}
			}
			else if (gApp.fieldView == DemoFieldView::Colorful)
			{
				// 功能 4 第 4 点：颜色鲜艳度（0=全白，1=全鲜艳，与时间/方向无关）
				float saturation = gApp.definition.colorfulSaturation;
				if (ImGui::SliderFloat(
					"Color saturation", &saturation, 0.0f, 1.0f, "%.2f"))
				{
					gApp.definition.colorfulSaturation = saturation;
				}
			}
			// jetWidth：入口几何由 flag 固化，必须重建（§4.1 第三类）。
			// 请求 4：仅 Jet flow 显示。滑块拖动过程中不触发任何动作，
			// 松开（IsItemDeactivated）且值确实变化时才写入 definition，
			// 并用 RestartWithCurrentSettings 按新 jetWidth 重启（保留其余设置）。
			if (gApp.caseId == DemoCaseId::JetFlow)
			{
				int jetWidth = gApp.definition.jetWidth;
				ImGui::SliderInt(
					"Jet width", &jetWidth, 2, std::max(2, gApp.definition.nx / 2));
				if (ImGui::IsItemDeactivated() &&
					jetWidth != gApp.definition.jetWidth)
				{
					gApp.definition.jetWidth = jetWidth;
					RestartWithCurrentSettings(window);
				}
			}
		}

		// 请求 4：色条不再作为面板组件，改由 DrawLegendOverlay 固定在
		// 画面右下角（Render 中调用，见下方实现）。

		// ---- OBJECTS ----
		if (ImGui::CollapsingHeader("OBJECTS", ImGuiTreeNodeFlags_DefaultOpen))
		{
			const bool objectsEnabled =
				gApp.definition.hasMovableObstacle;
			ImGui::BeginDisabled(!objectsEnabled);
			if (!objectsEnabled)
			{
				ImGui::TextWrapped("Objects are disabled for this case.");
			}
			else
			{
				const char* shapeItems[] = { "Circle", "Box", "Diamond" };
				int shapeIndex = (int)gApp.obstacleShape;
				if (ImGui::Combo("Shape", &shapeIndex, shapeItems, 3))
				{
					const ObstacleShape newShape = (ObstacleShape)shapeIndex;
					if (newShape != gApp.obstacleShape)
					{
						gApp.obstacleShape = newShape;
						gApp.SyncBodiesToFlow();   // §2.2 形状切换 → 全量重分类
					}
				}

				ImGui::Text("Bodies: %d / %d", gApp.bodyCount, kMaxBodies);
				if (ImGui::Button("Add body"))
				{
					// §2.5 候选位置：流场中心附近逐点尝试（与现有物体当前位置校验）
					const float radius = gApp.definition.obstacleRadius;
					const float cx = 0.5f * (float)(gApp.definition.nx - 1);
					const float cy = 0.5f * (float)(gApp.definition.ny - 1);
					const float offsets[][2] = {
						{ 0.0f, 0.0f }, { 0.0f, 25.0f }, { 0.0f, -25.0f },
						{ 25.0f, 0.0f }, { -25.0f, 0.0f },
						{ 0.0f, 50.0f }, { 0.0f, -50.0f }
					};
					bool added = false;
					for (const float* offset : offsets)
					{
						if (gApp.AddBodyAt(
							cx + offset[0], cy + offset[1], radius))
						{
							added = true;
							break;
						}
					}
					if (!added)
					{
						ImGui::OpenPopup("body_limit");
					}
				}
				if (ImGui::BeginPopup("body_limit"))
				{
					ImGui::Text("No free spot (limit or overlap).");
					ImGui::EndPopup();
				}

				// 物体单选列表
				for (int i = 0; i < gApp.bodyCount; i++)
				{
					char label[64];
					sprintf_s(
						label,
						"Body #%d  (r=%.1f)",
						i + 1,
						gApp.bodies[i].radius);
					if (ImGui::Selectable(
						label, gApp.bodies[i].selected))
					{
						gApp.SelectBody(i);
					}
				}

				// 选中物体控制（Center X/Y 写目标点 tx/ty，平滑移动）
				if (gApp.bodyCount > 0 &&
					gApp.selectedBody >= 0 &&
					gApp.selectedBody < gApp.bodyCount)
				{
					const int index = gApp.selectedBody;
					RigidBody& body = gApp.bodies[index];
					const float margin = body.radius + 2.0f;
					const float minX = margin;
					const float maxX = (float)gApp.definition.nx - margin - 1.0f;
					const float minY = margin;
					const float maxY = (float)gApp.definition.ny - margin - 1.0f;

					ImGui::Separator();
					float centerX = body.tx;
					if (ImGui::SliderFloat("Center X", &centerX, minX, maxX, "%.1f"))
					{
						body.tx = ClampFloat(centerX, minX, maxX);
					}
					float centerY = body.ty;
					if (ImGui::SliderFloat("Center Y", &centerY, minY, maxY, "%.1f"))
					{
						body.ty = ClampFloat(centerY, minY, maxY);
					}
					float radius = body.radius;
					if (ImGui::SliderFloat("Radius", &radius, 2.0f, 20.0f, "%.1f"))
					{
						body.radius = ClampFloat(radius, 2.0f, 20.0f);
						gApp.SyncBodiesToFlow();   // §2.2 半径变化 → 重分类
						gApp.SyncBodyToDefinition();   // §4.1 数据同源（Re 显示）
					}
					if (ImGui::Button("Remove selected"))
					{
						gApp.RemoveBody(index);
					}
					ImGui::Text("Force X: %.4f", gApp.bodyForceX[index]);
					ImGui::Text("Force Y: %.4f", gApp.bodyForceY[index]);
					ImGui::Text("Torque: %.4f", gApp.bodyTorque[index]);
				}
			}
			ImGui::EndDisabled();
		}

		// ---- TOOL（功能 4，通用工具栏：吹风 / 旋涡；点按确认式，仅两个选项直接展开）----
		if (ImGui::CollapsingHeader("Tool", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// 点按确认式工具按钮：点击选中，再次点击已选中的工具 → 取消（回到 None）。
			// 交互逻辑不变：选中后左键即用该工具；未选中时仍可用快捷键
			//（Shift+左键=吹风、中键=旋涡，见 ApplyMouseEffects）。
			const float toolWidth = 90.0f;
			if (ImGui::Selectable(
				"Blow",
				gApp.activeTool == LbmApp::InteractionTool::Blow,
				0,
				ImVec2(toolWidth, 0.0f)))
			{
				gApp.activeTool = (gApp.activeTool == LbmApp::InteractionTool::Blow)
					? LbmApp::InteractionTool::None
					: LbmApp::InteractionTool::Blow;
			}
			ImGui::SameLine();
			if (ImGui::Selectable(
				"Vortex",
				gApp.activeTool == LbmApp::InteractionTool::Vortex,
				0,
				ImVec2(toolWidth, 0.0f)))
			{
				gApp.activeTool = (gApp.activeTool == LbmApp::InteractionTool::Vortex)
					? LbmApp::InteractionTool::None
					: LbmApp::InteractionTool::Vortex;
			}
			ImGui::SliderFloat("Brush radius", &gApp.brushRadius, 1.0f, 20.0f);
			ImGui::SliderFloat("Blow strength", &gApp.blowStrength, 0.0f, 0.02f, "%.4f");
		}

		// ---- DEBUG（功能 5 预留，§5.3）----
		if (ImGui::CollapsingHeader("DEBUG (reserved)"))
		{
			ImGui::TextWrapped("Hold Ctrl + click a cell to inspect (M6).");
			ImGui::Checkbox("Debug overlay", &gApp.debugMode);
		}

		// ---- CONTROLS ----
		if (ImGui::CollapsingHeader("CONTROLS"))
		{
			ImGui::TextWrapped(
				"1/2 case | V view | Space pause | S step | +/- steps\n"
				"R reset | Tab select | Arrows move | Esc quit");
		}

		ImGui::End();
	}

	void Render(HWND window)
	{
		if (!gApp.initialized || gDeviceContext == NULL)
		{
			return;
		}

		RECT client;
		GetClientRect(window, &client);
		const int width = client.right - client.left;
		const int height = client.bottom - client.top;
		if (width <= 0 || height <= 0)
		{
			return;
		}

		// §5.1-4：DisplaySize 取物理像素，不设 DisplayFramebufferScale（默认 1）
		static DWORD lastFrameTick = 0;
		const DWORD nowTick = GetTickCount();
		ImGuiIO& io = ImGui::GetIO();
		io.DisplaySize = ImVec2((float)width, (float)height);
		if (lastFrameTick == 0)
		{
			lastFrameTick = nowTick;
		}
		io.DeltaTime = std::max(0.001f, (float)(nowTick - lastFrameTick) / 1000.0f);
		lastFrameTick = nowTick;

		glViewport(0, 0, width, height);
		glClearColor(0.035f, 0.043f, 0.051f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		BeginScreenCoordinates(width, height);

		const UiLayout layout = ComputeLayout(width, height);
		DrawField(layout.field);
		DrawBodies(layout.field);
		DrawDebugOverlay(layout);
		// 功能 4 第 3 点：colorful 色条随时间旋转，需每帧重建（256×1 纹理，开销可忽略）
		if (gApp.fieldView == DemoFieldView::Colorful)
		{
			RebuildLegendTexture();
		}
		DrawLegendOverlay(layout.field);   // 请求 4：色条固定在画面右下角

		// §5.2 新渲染顺序：ImGui 帧在场景绘制之后
		ImGui_ImplOpenGL2_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		BuildAppUi(window, layout);
		ImGui::Render();
		ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

		SwapBuffers(gDeviceContext);
	}

	void ResetSimulation(HWND window, DemoCaseId caseId)
	{
		gApp.Release();
		gApp.Init(caseId);
		CreateFieldTexture();
		UpdateWindowTitle(window);
		InvalidateRect(window, NULL, FALSE);
	}

	void AdvanceOneUiFrame()
	{
		gApp.AdvanceObstacles();
		gApp.Step();
	}

	// §4.3 启动加载分发：把 gLoadedParams 的参数按 §4.1 三分类应用到当前算例。
	// 必须在 InitOpenGL 之后调用（重建路径需要 GL 上下文创建纹理）。
	// 也复用于面板 [Load ini] 与预设应用（§4.2 切换预设 → def 拷贝 + 三分类分发）。
	// 请求 2：forceRebuild=true 时无条件重建（Restart：以当前设置重新开始）。
	void ApplyLoadedStartupParams(HWND window, bool forceRebuild)
	{
		if (!gLoadedParams.valid || !gApp.initialized)
		{
			return;
		}

		const DemoCaseDefinition& loaded = gLoadedParams.def;
		// §4.1 第三类（需重建）：网格尺寸 / 初始条件 / 入口几何（flag 固化）
		const bool needsRebuild = forceRebuild ||
			loaded.nx != gApp.definition.nx ||
			loaded.ny != gApp.definition.ny ||
			loaded.initialUx != gApp.definition.initialUx ||
			loaded.initialUy != gApp.definition.initialUy ||
			loaded.jetWidth != gApp.definition.jetWidth;

		if (needsRebuild)
		{
			gApp.Release();
			gApp.Init(gLoadedParams.caseId);
		}

		// 覆盖 definition（重建后 Init 已写入该 case 的默认值，这里写回加载值）
		gApp.definition.viscosity = loaded.viscosity;
		gApp.definition.initialUx = loaded.initialUx;
		gApp.definition.initialUy = loaded.initialUy;
		gApp.definition.inletUx = loaded.inletUx;
		gApp.definition.inletUy = loaded.inletUy;
		gApp.definition.inletPerturbationAmplitude =
			loaded.inletPerturbationAmplitude;
		gApp.definition.inletPerturbationPeriod =
			loaded.inletPerturbationPeriod;
		gApp.definition.speedColorMax = loaded.speedColorMax;
		gApp.definition.vorticityColorMax = loaded.vorticityColorMax;
		gApp.definition.colorfulSaturation = loaded.colorfulSaturation;
		gApp.definition.jetWidth = loaded.jetWidth;

		gApp.fieldView = gLoadedParams.view;
		gApp.stepsPerFrame = gLoadedParams.stepsPerFrame;
		gApp.obstacleShape = gLoadedParams.shape;

		// 物体系统：§1.2 约定 hasMovableObstacle == false（Jet）时 bodyCount == 0
		gApp.bodyCount = gApp.definition.hasMovableObstacle
			? gLoadedParams.bodyCount
			: 0;
		for (int i = 0; i < gApp.bodyCount; i++)
		{
			gApp.bodies[i] = gLoadedParams.bodies[i];
			gApp.bodies[i].vx = 0.0f;   // 刚载入不移动
			gApp.bodies[i].vy = 0.0f;
		}
		if (gApp.bodyCount > 0)
		{
			gApp.SelectBody(0);
			gApp.SyncBodyToDefinition();   // §4.1 数据同源（Re 显示）
		}
		else
		{
			gApp.selectedBody = -1;
		}

		if (needsRebuild)
		{
			// §4.1 第三类字段在 Init 中取默认值：按加载值重新初始化流场，
			// 使 initialUx/Uy 与 jetWidth（入口几何）真正生效；物体姿态已在
			// bodies 中，一并写入固体格点。
			InitializeDemoCase(
				gApp.flow,
				gApp.definition,
				gApp.bodies.data(),
				gApp.bodyCount,
				gApp.obstacleShape);
			gApp.solver.mlTransData2Gpu();   // 全量上传（含粘度）
			BuildFieldImage(
				gApp.flow, gApp.definition, gApp.fieldView, gApp.pixels);
			CreateFieldTexture();
			UpdateWindowTitle(window);
		}
		else
		{
			// §4.1 第二类：粘度热更新（重建路径由 mlTransData2Gpu 覆盖）
			if (gApp.flow != NULL)
			{
				gApp.flow->vis_shear = gApp.definition.viscosity;
				SyncFlowScalarsToDevice(gApp.solver, 0);
			}
			// §2.2 物体姿态/数量/形状变化 → 全量重分类（含固体格点与壁面矩）
			gApp.SyncBodiesToFlow();
		}

		RebuildLegendTexture();   // §5.5 色条内容随 fieldView
		InvalidateRect(window, NULL, FALSE);
	}

	// 请求 1：UI 大小模式切换。小 = 当前默认窗口（1040×820）；
	// 中 = 按工作区 75%×85% 取中等尺寸；大 = 全屏（去边框覆盖整个显示器）。
	void ApplyUiSizeMode(HWND window, UiSizeMode mode)
	{
		gUiSizeMode = mode;

		// 请求 3：文字与控件（按钮/滑块等）尺寸随模式等比缩放。
		// 先恢复基准样式再整体缩放，避免多次切换时缩放系数累积。
		static ImGuiStyle sBaseStyle;
		static bool sBaseStyleReady = false;
		if (!sBaseStyleReady)
		{
			sBaseStyle = ImGui::GetStyle();
			sBaseStyleReady = true;
		}
		ImGui::GetStyle() = sBaseStyle;
		ImGui::GetStyle().ScaleAllSizes(UiScale());
		ImGui::GetIO().FontGlobalScale = UiScale();

		HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
		MONITORINFO monitorInfo = {};
		monitorInfo.cbSize = sizeof(monitorInfo);
		GetMonitorInfo(monitor, &monitorInfo);
		const RECT& full = monitorInfo.rcMonitor;
		const RECT& work = monitorInfo.rcWork;

		if (mode == UiSizeMode::Large)
		{
			// 全屏：去掉非客户区（标题栏/边框）
			SetWindowLongPtrW(
				window, GWL_STYLE, (LONG_PTR)(WS_POPUP | WS_VISIBLE));
			SetWindowPos(
				window, HWND_TOP,
				full.left, full.top,
				full.right - full.left, full.bottom - full.top,
				SWP_FRAMECHANGED | SWP_SHOWWINDOW);
			return;
		}

		// 非全屏：恢复带边框样式，窗口居中
		SetWindowLongPtrW(
			window, GWL_STYLE,
			(LONG_PTR)(WS_OVERLAPPEDWINDOW | WS_VISIBLE));
		int newWidth = 1040;   // 小 = 启动默认
		int newHeight = 820;
		if (mode == UiSizeMode::Medium)
		{
			newWidth = (int)((float)(work.right - work.left) * 0.75f);
			newHeight = (int)((float)(work.bottom - work.top) * 0.85f);
		}
		const int centerX =
			work.left + (work.right - work.left - newWidth) / 2;
		const int centerY =
			work.top + (work.bottom - work.top - newHeight) / 2;
		SetWindowPos(
			window, HWND_TOP, centerX, centerY, newWidth, newHeight,
			SWP_FRAMECHANGED | SWP_SHOWWINDOW);
	}

	// 请求 2/4：以当前全部设置重建算例（不改动已调参数）。
	// 供 Restart 按钮与 jet width 应用共用；解决 jet width 调整后被
	// ResetSimulation 以默认值覆盖（更改丢失）的 bug。
	void RestartWithCurrentSettings(HWND window)
	{
		LoadedStartupParams saved;
		saved.caseId = gApp.caseId;
		saved.def = gApp.definition;
		saved.view = gApp.fieldView;
		saved.stepsPerFrame = gApp.stepsPerFrame;
		saved.shape = gApp.obstacleShape;
		saved.bodyCount = gApp.bodyCount;
		saved.bodies = gApp.bodies;
		saved.valid = true;
		gLoadedParams = saved;
		ApplyLoadedStartupParams(window, true);   // forceRebuild
		UpdateWindowTitle(window);
	}

	// §4.2 预设系统：内置 Karman / Jet + 运行时 Custom（功能 4：smoke 预设已移除）。
	// 切换预设 → def 拷贝 → 复用 ApplyLoadedStartupParams 分发。
	void ApplyPreset(HWND window, int presetIndex)
	{
		static const struct
		{
			const char* name;
			DemoCaseId id;
		} kPresets[] =
		{
			{ "Karman", DemoCaseId::KarmanVortex },
			{ "Jet", DemoCaseId::JetFlow },
			{ "Custom", DemoCaseId::KarmanVortex }   // 运行时 custom：当前状态
		};
		if (presetIndex < 0 || presetIndex >= (int)(sizeof(kPresets) / sizeof(kPresets[0])))
		{
			return;
		}

		gActivePreset = presetIndex;
		if (presetIndex == 2)
		{
			return;   // Custom：不做任何修改，仅标记
		}

		const DemoCaseDefinition def = GetDefaultDefinition(kPresets[presetIndex].id);
		gLoadedParams.valid = true;
		gLoadedParams.caseId = kPresets[presetIndex].id;
		gLoadedParams.def = def;
		gLoadedParams.view = def.defaultView;
		gLoadedParams.stepsPerFrame = def.initialStepsPerFrame;
		gLoadedParams.shape = ObstacleShape::Circle;
		gLoadedParams.bodyCount = def.hasMovableObstacle ? 1 : 0;
		gLoadedParams.bodies = {};
		if (gLoadedParams.bodyCount > 0)
		{
			gLoadedParams.bodies[0] = {
				def.obstacleStartX, def.obstacleStartY,
				0.0f, 0.0f,
				def.obstacleStartX, def.obstacleStartY,
				def.obstacleRadius, true
			};
		}

		ApplyLoadedStartupParams(window);
		UpdateWindowTitle(window);
	}

	LRESULT CALLBACK WindowProcedure(
		HWND window,
		UINT message,
		WPARAM wParam,
		LPARAM lParam)
	{
		switch (message)
		{
		case WM_CREATE:
			gApp.Init(gStartupCase);
			if (!InitOpenGL(window))
			{
				MessageBoxW(
					window,
					L"OpenGL context creation failed.",
					L"Home2D LBM",
					MB_ICONERROR);
				return -1;
			}
			// §4.3 启动加载分发（须在 InitOpenGL 之后：重建路径需要 GL 上下文）
			ApplyLoadedStartupParams(window);
			UpdateWindowTitle(window);
			SetTimer(window, 1, kTimerMilliseconds, NULL);
			return 0;

		case WM_TIMER:
			gApp.UpdateObstacleTarget();
			if (!gApp.paused)
			{
				AdvanceOneUiFrame();
				UpdateWindowTitle(window);
			}
			InvalidateRect(window, NULL, FALSE);
			return 0;

		// §1.3 / §3.1 鼠标输入系统：命中拾取 + 拖动（SetCapture 防拖出客户区卡死）
		case WM_LBUTTONDOWN:
		{
			// §5.1-5 消息转发：先喂给 ImGui，再按 WantCaptureMouse 分支
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			const UiLayout layout = CurrentLayout(window);
			const int mouseX = GET_X_LPARAM(lParam);
			const int mouseY = GET_Y_LPARAM(lParam);
			gApp.mouseScreenX = mouseX;
			gApp.mouseScreenY = mouseY;
			gApp.lButtonDown = true;
			ScreenToFieldPoint(mouseX, mouseY, layout, gApp.mouseFieldX, gApp.mouseFieldY);

			// §5.4 面板豁免：ImGui 捕获 + panel 矩形双保险
			if (IsOverUiPanel(mouseX, mouseY, layout) || UiWantsCaptureMouse())
			{
				gApp.dragging = false;
				return 0;
			}

			// §3.2 命中拾取：从后往前遍历，后添加者优先
			const int hit = gApp.PickBody(gApp.mouseFieldX, gApp.mouseFieldY);
			if (hit >= 0)
			{
				gApp.SelectBody(hit);
				gApp.dragging = true;
				gApp.grabOffsetX = gApp.bodies[hit].x - gApp.mouseFieldX;
				gApp.grabOffsetY = gApp.bodies[hit].y - gApp.mouseFieldY;
				SetCapture(window);   // §3.1-补足 1：拖出客户区仍能收到 WM_MOUSEMOVE
			}
			else
			{
				gApp.dragging = false;
			}
			return 0;
		}

		case WM_MOUSEMOVE:
		{
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			const UiLayout layout = CurrentLayout(window);
			const int mouseX = GET_X_LPARAM(lParam);
			const int mouseY = GET_Y_LPARAM(lParam);
			gApp.mouseScreenX = mouseX;
			gApp.mouseScreenY = mouseY;
			ScreenToFieldPoint(mouseX, mouseY, layout, gApp.mouseFieldX, gApp.mouseFieldY);

			// §3.1 进行中的拖动：每帧更新目标，clamp 边界沿用 margin = radius + 2
			if (gApp.dragging &&
				gApp.selectedBody >= 0 &&
				gApp.selectedBody < gApp.bodyCount &&
				!UiWantsCaptureMouse())
			{
				const RigidBody& body = gApp.bodies[gApp.selectedBody];
				const float margin = body.radius + 2.0f;
				gApp.bodies[gApp.selectedBody].tx = ClampFloat(
					gApp.mouseFieldX + gApp.grabOffsetX,
					margin,
					(float)gApp.definition.nx - margin - 1.0f);
				gApp.bodies[gApp.selectedBody].ty = ClampFloat(
					gApp.mouseFieldY + gApp.grabOffsetY,
					margin,
					(float)gApp.definition.ny - margin - 1.0f);
			}
			return 0;
		}

		case WM_LBUTTONUP:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			gApp.lButtonDown = false;
			gApp.dragging = false;
			ReleaseCapture();   // §3.1-补足 1：物体保持当前位置，高亮保留
			return 0;

		// §1.3 功能 4/5 预留消息：本期仅转发 ImGui
		case WM_RBUTTONDOWN:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			gApp.rButtonDown = true;
			return 0;

		case WM_RBUTTONUP:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			gApp.rButtonDown = false;
			return 0;

		case WM_MBUTTONDOWN:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			gApp.mButtonDown = true;
			return 0;

		case WM_MBUTTONUP:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			gApp.mButtonDown = false;
			return 0;

		case WM_MOUSEWHEEL:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			return 0;

		case WM_KEYDOWN:
		{
			// §5.1-5 消息转发：先喂给 ImGui
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			// §5.4 快捷键让位：ImGui 捕获键盘（数值输入框等）时跳过全部应用快捷键
			if (UiWantsCaptureKeyboard())
			{
				return 0;
			}
			const bool wasDown = (lParam & (1LL << 30)) != 0;
			const bool repeatableCommand =
				wParam == 'S' ||
				wParam == VK_OEM_PLUS ||
				wParam == VK_ADD ||
				wParam == VK_OEM_MINUS ||
				wParam == VK_SUBTRACT;
			if (wParam == VK_LEFT ||
				wParam == VK_RIGHT ||
				wParam == VK_UP ||
				wParam == VK_DOWN)
			{
				gApp.SetMoveKey(wParam, true);
				return 0;
			}
			if (wasDown && !repeatableCommand)
			{
				return 0;
			}
			if (wParam == VK_ESCAPE)
			{
				DestroyWindow(window);
			}
			else if (wParam == '1')
			{
				ResetSimulation(window, DemoCaseId::KarmanVortex);
			}
			else if (wParam == '2')
			{
				ResetSimulation(window, DemoCaseId::JetFlow);
			}
			else if (wParam == 'V')
			{
				gApp.ToggleFieldView();
				RebuildLegendTexture();   // §5.5 色条内容随视图切换
				UpdateWindowTitle(window);
				InvalidateRect(window, NULL, FALSE);
			}
			else if (wParam == VK_SPACE)
			{
				gApp.paused = !gApp.paused;
				UpdateWindowTitle(window);
			}
			else if (wParam == 'S')
			{
				AdvanceOneUiFrame();
				UpdateWindowTitle(window);
				InvalidateRect(window, NULL, FALSE);
			}
			else if (wParam == 'R')
			{
				ResetSimulation(window, gApp.caseId);
			}
			else if (wParam == VK_TAB)
			{
				// §2.5 Tab 循环切换选中物体（按住不重复循环：被 wasDown 检查拦截）
				gApp.CycleSelection();
			}
			else if (wParam == VK_OEM_PLUS ||
				wParam == VK_ADD)
			{
				gApp.stepsPerFrame =
					std::min(
						kMaximumStepsPerFrame,
						gApp.stepsPerFrame + 1);
				UpdateWindowTitle(window);
			}
			else if (wParam == VK_OEM_MINUS || wParam == VK_SUBTRACT)
			{
				gApp.stepsPerFrame =
					std::max(
						kMinimumStepsPerFrame,
						gApp.stepsPerFrame - 1);
				UpdateWindowTitle(window);
			}
			return 0;
		}

		case WM_KEYUP:
			ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam);
			if (!UiWantsCaptureKeyboard())
			{
				if (wParam == VK_LEFT ||
					wParam == VK_RIGHT ||
					wParam == VK_UP ||
					wParam == VK_DOWN)
				{
					gApp.SetMoveKey(wParam, false);
				}
			}
			return 0;

		case WM_KILLFOCUS:
			gApp.ClearMoveKeys();
			gApp.dragging = false;
			ReleaseCapture();   // §3.1：防止鼠标松在客户区外的残留拖动状态
			return 0;

		case WM_GETMINMAXINFO:
		{
			MINMAXINFO* information = (MINMAXINFO*)lParam;
			information->ptMinTrackSize.x = 820;
			information->ptMinTrackSize.y = 650;
			return 0;
		}

		case WM_ERASEBKGND:
			return 1;

		case WM_PAINT:
		{
			PAINTSTRUCT paint;
			BeginPaint(window, &paint);
			Render(window);
			EndPaint(window, &paint);
			return 0;
		}

		case WM_SIZE:
			InvalidateRect(window, NULL, FALSE);
			return 0;

		case WM_DESTROY:
			KillTimer(window, 1);
			gApp.Release();
			ShutdownOpenGL(window);
			PostQuitMessage(0);
			return 0;
		}

		return DefWindowProc(window, message, wParam, lParam);
	}

	bool ParseArguments(int argc, char** argv)
	{
		for (int i = 1; i < argc; i++)
		{
			const std::string argument = argv[i];
			if (argument == "--help" || argument == "-h")
			{
				std::cout
					<< "Usage: LBM_MRLBM.exe [--case karman|jetflow]\n";
				gHelpRequested = true;
				return false;
			}

			std::string caseName;
			if (argument == "--case" && i + 1 < argc)
			{
				caseName = argv[++i];
			}
			else if (argument.find("--case=") == 0)
			{
				caseName = argument.substr(7);
			}
			else
			{
				std::cerr << "Unknown argument: " << argument << "\n";
				return false;
			}

			if (!ParseDemoCaseName(caseName, gStartupCase))
			{
				std::cerr << "Unknown case: " << caseName << "\n";
				return false;
			}
			// §4.3 优先级：命令行显式 --case 时，ini 不覆盖 case（只覆盖其余参数）
			gCaseSpecifiedOnCommandLine = true;
		}
		return true;
	}
}

int main(int argc, char** argv)
{
	if (!ParseArguments(argc, argv))
	{
		return gHelpRequested ? 0 : 1;
	}

	// §4.3 启动加载：读 params.ini 存入模块级 gLoadedParams，WM_CREATE 后分发。
	// 优先级约定（写死）：命令行显式 --case 时 ini 不覆盖 case（applyFileCase=false）。
	gLoadedParams.caseId = gStartupCase;
	gLoadedParams.valid = LoadAppParams(
		"params.ini",
		gLoadedParams.caseId,
		gLoadedParams.def,
		gLoadedParams.view,
		gLoadedParams.stepsPerFrame,
		gLoadedParams.shape,
		gLoadedParams.bodyCount,
		gLoadedParams.bodies.data(),
		!gCaseSpecifiedOnCommandLine);
	if (gLoadedParams.valid)
	{
		std::cout << "Loaded params.ini (case="
			<< (gLoadedParams.def.cliName != nullptr
				? gLoadedParams.def.cliName
				: "unknown")
			<< ").\n";
	}

	SetProcessDPIAware();
	HINSTANCE instance = GetModuleHandleW(NULL);
	const wchar_t* className = L"Home2DLbmOpenGLWindow";
	WNDCLASSW windowClass = {};
	windowClass.style = CS_OWNDC;
	windowClass.lpfnWndProc = WindowProcedure;
	windowClass.hInstance = instance;
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.hbrBackground = NULL;
	windowClass.lpszClassName = className;

	if (!RegisterClassW(&windowClass))
	{
		MessageBoxW(
			NULL,
			L"Window class registration failed.",
			L"Home2D LBM",
			MB_ICONERROR);
		return 1;
	}

	HWND window = CreateWindowExW(
		0,
		className,
		L"Home2D LBM",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		1040,
		820,
		NULL,
		NULL,
		instance,
		NULL);
	if (window == NULL)
	{
		MessageBoxW(
			NULL,
			L"Window creation failed.",
			L"Home2D LBM",
			MB_ICONERROR);
		return 1;
	}
	// 功能 4：窗口句柄交给 gApp（hover 豁免现算 layout 需要，§5.4/§10.12）
	gApp.appWindow = window;

	MSG message;
	while (GetMessageW(&message, NULL, 0, 0) > 0)
	{
		TranslateMessage(&message);
		DispatchMessageW(&message);
	}
	return (int)message.wParam;
}
