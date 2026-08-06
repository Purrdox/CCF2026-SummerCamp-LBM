#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "LbmCases2D.h"
#include "../inc/2D/cpu/mrSolver2D.h"

#include <windows.h>
#include <gl/GL.h>

#include <algorithm>
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
	const float kMaximumWallSpeed = 0.06f;
	const int kInverseDirection[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

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

	struct LbmApp
	{
		mrSolver2D solver;
		mrFlow2D* flow = NULL;
		std::vector<mrFlow2D*> flows;
		std::vector<mrFlow2D*> deviceFlows;
		std::vector<unsigned char> pixels;
		const DemoCaseDefinition* definition = NULL;
		DemoCaseId caseId = DemoCaseId::KarmanVortex;
		DemoFieldView fieldView = DemoFieldView::Vorticity;
		int frame = 0;
		int iteration = 0;
		int stepsPerFrame = 5;
		float obstacleX = 0.0f;
		float obstacleY = 0.0f;
		float obstacleTargetX = 0.0f;
		float obstacleTargetY = 0.0f;
		float obstacleVelocityX = 0.0f;
		float obstacleVelocityY = 0.0f;
		float solidForceX = 0.0f;
		float solidForceY = 0.0f;
		float solidTorque = 0.0f;
		bool hasForceSample = false;
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
			definition = NULL;
			initialized = false;
		}

		void Init(DemoCaseId newCase)
		{
			caseId = newCase;
			definition = &GetDemoCaseDefinition(caseId);
			fieldView = definition->defaultView;
			stepsPerFrame = definition->initialStepsPerFrame;
			frame = 0;
			iteration = 0;
			paused = false;
			hasForceSample = false;
			solidForceX = 0.0f;
			solidForceY = 0.0f;
			solidTorque = 0.0f;
			obstacleX = definition->obstacleStartX;
			obstacleY = definition->obstacleStartY;
			obstacleTargetX = obstacleX;
			obstacleTargetY = obstacleY;
			obstacleVelocityX = 0.0f;
			obstacleVelocityY = 0.0f;
			ClearMoveKeys();

			flow = new mrFlow2D();
			flow->Create(
				0.0f,
				0.0f,
				definition->nx,
				definition->ny,
				1.0f,
				(REAL)definition->nx,
				(REAL)definition->ny,
				definition->viscosity,
				0.0f);

			InitializeDemoCase(
				flow, *definition, obstacleX, obstacleY);

			flows.push_back(flow);
			deviceFlows.push_back(NULL);
			solver.gpuId = 0;
			solver.L = 1.0f;
			solver.AttachLbmHost(flows);
			solver.AttachLbmDevice(deviceFlows);
			solver.mlTransData2Gpu();

			BuildFieldImage(flow, *definition, fieldView, pixels);
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
			if (!initialized || !definition->hasMovableObstacle)
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

			const float length = sqrtf(dx * dx + dy * dy);
			dx = dx / length * definition->obstacleTargetSpeed;
			dy = dy / length * definition->obstacleTargetSpeed;

			const float margin = definition->obstacleRadius + 2.0f;
			obstacleTargetX = ClampFloat(
				obstacleTargetX + dx,
				margin,
				(float)definition->nx - margin - 1.0f);
			obstacleTargetY = ClampFloat(
				obstacleTargetY + dy,
				margin,
				(float)definition->ny - margin - 1.0f);

			const float leadX = obstacleTargetX - obstacleX;
			const float leadY = obstacleTargetY - obstacleY;
			const float lead = sqrtf(leadX * leadX + leadY * leadY);
			if (lead > kMaximumTargetLead)
			{
				obstacleTargetX = obstacleX + leadX / lead * kMaximumTargetLead;
				obstacleTargetY = obstacleY + leadY / lead * kMaximumTargetLead;
			}
		}

		void ApplyObstaclePose(
			float newX,
			float newY,
			float wallUx,
			float wallUy)
		{
			const int nx = definition->nx;
			const int ny = definition->ny;
			const int count = nx * ny;
			std::vector<MLLATTICENODE_FLAG> oldFlags(
				flow->flag, flow->flag + count);
			std::vector<MLLATTICENODE_FLAG> newFlags(count);
			std::vector<int> releasedCells;
			std::vector<int> solidCells;
			std::vector<int> touchedCells;
			bool flagsChanged = false;

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
							GetDemoCaseBaseFlag(*definition, x, y);
						const bool isSolid = IsDemoCaseObstacleCell(
							*definition, x, y, newX, newY);
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
				for (const int solidIdx : solidCells)
				{
					const int sx = solidIdx % nx;
					const int sy = solidIdx / nx;
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
								solidIdx, neighbor, wallUx, wallUy);
							found = true;
							break;
						}
					}
					if (!found)
					{
						WriteEquilibriumMoments2D(
							flow, solidIdx, 1.0f, wallUx, wallUy);
					}
				}

				// 4. Record every touched cell and update the obstacle pose.
				touchedCells.insert(touchedCells.end(),
					solidCells.begin(), solidCells.end());
				touchedCells.insert(touchedCells.end(),
					releasedCells.begin(), releasedCells.end());

				obstacleX = newX;
				obstacleY = newY;
				obstacleVelocityX = wallUx;
				obstacleVelocityY = wallUy;

				for (int idx = 0; idx < count; idx++)
				{
					flow->flag[idx] = newFlags[idx];
				}
				flagsChanged = true;
			}

			// ===== ASSIGNMENT FILL END: P2-B =====
			SyncCellsToDevice(solver, 0, touchedCells, flagsChanged);
		}

		void AdvanceObstacle()
		{
			if (!initialized || !definition->hasMovableObstacle)
			{
				return;
			}

			const float dx = obstacleTargetX - obstacleX;
			const float dy = obstacleTargetY - obstacleY;
			const float distance = sqrtf(dx * dx + dy * dy);
			if (distance <= 1.0e-5f)
			{
				if (fabsf(obstacleVelocityX) > 1.0e-6f ||
					fabsf(obstacleVelocityY) > 1.0e-6f)
				{
					ApplyObstaclePose(obstacleX, obstacleY, 0.0f, 0.0f);
				}
				return;
			}

			const float maxDisplacement = std::min(
				definition->obstacleMoveSpeed,
				kMaximumWallSpeed * (float)stepsPerFrame);
			const float displacement = std::min(distance, maxDisplacement);
			const float moveX = dx / distance * displacement;
			const float moveY = dy / distance * displacement;
			ApplyObstaclePose(
				obstacleX + moveX,
				obstacleY + moveY,
				moveX / (float)stepsPerFrame,
				moveY / (float)stepsPerFrame);
		}

		void ComputeSolidLoads()
		{
			if (!definition->hasMovableObstacle)
			{
				solidForceX = 0.0f;
				solidForceY = 0.0f;
				solidTorque = 0.0f;
				hasForceSample = false;
				return;
			}

			REAL rawForceX = 0.0f;
			REAL rawForceY = 0.0f;
			REAL rawTorque = 0.0f;
			const int nx = definition->nx;
			const int ny = definition->ny;

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

						rawForceX += forceX;
						rawForceY += forceY;
						rawTorque +=
							(boundaryX - obstacleX) * forceY -
							(boundaryY - obstacleY) * forceX;
					}
				}
			}

			const float blend = hasForceSample ? 0.15f : 1.0f;
			solidForceX += blend * ((float)rawForceX - solidForceX);
			solidForceY += blend * ((float)rawForceY - solidForceY);
			solidTorque += blend * ((float)rawTorque - solidTorque);
			hasForceSample = true;
		}

		void Step()
		{
			if (!initialized)
			{
				return;
			}

			RefreshDemoCaseBoundaries(
				flow, *definition, iteration);
			SyncOuterBoundaryMomentsToDevice(solver, 0);

			for (int step = 0; step < stepsPerFrame; step++)
			{
				solver.mlIterateGpu();
				iteration++;
			}

			CopyMomentsFromDevice(solver, 0);
			ComputeSolidLoads();
			BuildFieldImage(flow, *definition, fieldView, pixels);
			frame++;
		}

		void ToggleFieldView()
		{
			fieldView = fieldView == DemoFieldView::VelocityMagnitude
				? DemoFieldView::Vorticity
				: DemoFieldView::VelocityMagnitude;
			if (initialized)
			{
				BuildFieldImage(flow, *definition, fieldView, pixels);
			}
		}
	};

	LbmApp gApp;
	DemoCaseId gStartupCase = DemoCaseId::KarmanVortex;
	bool gHelpRequested = false;
	HDC gDeviceContext = NULL;
	HGLRC gOpenGlContext = NULL;
	GLuint gFieldTexture = 0;
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
			gApp.definition->nx,
			gApp.definition->ny,
			0,
			GL_RGB,
			GL_UNSIGNED_BYTE,
			gApp.pixels.data());
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
		return CreateFontLists();
	}

	void ShutdownOpenGL(HWND window)
	{
		if (gFieldTexture != 0)
		{
			glDeleteTextures(1, &gFieldTexture);
			gFieldTexture = 0;
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
		const float panelWidth =
			(float)std::min(kPanelWidth, std::max(300, width / 3));
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
			(float)gApp.definition->nx / (float)gApp.definition->ny;

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
			gApp.definition->nx,
			gApp.definition->ny,
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

	void DrawObstacleTarget(const RectF& field)
	{
		if (!gApp.definition->hasMovableObstacle)
		{
			return;
		}

		const float x = field.left +
			gApp.obstacleTargetX / (float)(gApp.definition->nx - 1) *
			field.Width();
		const float y = field.bottom -
			gApp.obstacleTargetY / (float)(gApp.definition->ny - 1) *
			field.Height();
		const float radius =
			gApp.definition->obstacleRadius /
			(float)(gApp.definition->nx - 1) *
			field.Width();

		glColor3f(0.23f, 0.80f, 0.82f);
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < 48; i++)
		{
			const float angle =
				6.28318530717958647692f * (float)i / 48.0f;
			glVertex2f(
				x + cosf(angle) * radius,
				y + sinf(angle) * radius);
		}
		glEnd();
		DrawLine(x - 5.0f, y, x + 5.0f, y, 0.23f, 0.80f, 0.82f);
		DrawLine(x, y - 5.0f, x, y + 5.0f, 0.23f, 0.80f, 0.82f);
	}

	void DrawLegend(const RectF& panel, float y)
	{
		const float left = panel.left + 24.0f;
		const float right = panel.right - 24.0f;
		const float top = y + 8.0f;
		const float bottom = top + 13.0f;
		const int segments = 80;
		ColorRamp colorRamp;

		glBegin(GL_QUADS);
		for (int i = 0; i < segments; i++)
		{
			const float t0 = (float)i / (float)segments;
			const float t1 = (float)(i + 1) / (float)segments;
			float r0;
			float g0;
			float b0;
			float r1;
			float g1;
			float b1;
			if (gApp.fieldView == DemoFieldView::Vorticity)
			{
				MapVorticityColor(2.0f * t0 - 1.0f, r0, g0, b0);
				MapVorticityColor(2.0f * t1 - 1.0f, r1, g1, b1);
			}
			else
			{
				vec3 c0(0.0f, 0.0f, 0.0f);
				vec3 c1(0.0f, 0.0f, 0.0f);
				colorRamp.set_GLcolor(t0, COLOR__MAGMA, c0, false);
				colorRamp.set_GLcolor(t1, COLOR__MAGMA, c1, false);
				r0 = c0.x;
				g0 = c0.y;
				b0 = c0.z;
				r1 = c1.x;
				g1 = c1.y;
				b1 = c1.z;
			}

			const float x0 = left + (right - left) * t0;
			const float x1 = left + (right - left) * t1;
			glColor3f(r0, g0, b0);
			glVertex2f(x0, top);
			glVertex2f(x0, bottom);
			glColor3f(r1, g1, b1);
			glVertex2f(x1, bottom);
			glVertex2f(x1, top);
		}
		glEnd();

		glColor3f(0.64f, 0.68f, 0.72f);
		if (gApp.fieldView == DemoFieldView::Vorticity)
		{
			std::ostringstream negative;
			std::ostringstream positive;
			negative << "-" << std::fixed << std::setprecision(3)
				<< gApp.definition->vorticityColorMax;
			positive << "+" << std::fixed << std::setprecision(3)
				<< gApp.definition->vorticityColorMax;
			DrawText(left, bottom + 17.0f, negative.str());
			DrawText(0.5f * (left + right) - 4.0f, bottom + 17.0f, "0");
			DrawText(right - 50.0f, bottom + 17.0f, positive.str());
		}
		else
		{
			std::ostringstream maximum;
			maximum << std::fixed << std::setprecision(2)
				<< gApp.definition->speedColorMax;
			DrawText(left, bottom + 17.0f, "0");
			DrawText(right - 34.0f, bottom + 17.0f, maximum.str());
		}
	}

	void DrawPanel(const RectF& panel)
	{
		DrawSolidRect(panel, 0.075f, 0.086f, 0.098f, 1.0f);
		DrawLine(
			panel.left,
			panel.top,
			panel.left,
			panel.bottom,
			0.23f,
			0.26f,
			0.29f);

		const float left = panel.left + 24.0f;
		const float right = panel.right - 24.0f;
		float y = 31.0f;

		glColor3f(0.92f, 0.95f, 0.97f);
		DrawText(left, y, "HOME2D / D2Q9");
		y += 27.0f;
		glColor3f(0.25f, 0.81f, 0.82f);
		DrawText(left, y, gApp.definition->displayName);
		y += 20.0f;
		glColor3f(0.66f, 0.70f, 0.74f);
		DrawText(left, y, gApp.definition->description);
		y += 23.0f;
		DrawLine(left, y, right, y, 0.22f, 0.25f, 0.28f);
		y += 24.0f;

		glColor3f(0.48f, 0.72f, 0.92f);
		DrawText(left, y, "SIMULATION");
		y += 20.0f;
		glColor3f(0.82f, 0.84f, 0.86f);
		DrawText(left, y, gApp.paused ? "Status       Paused" : "Status       Running");
		y += 18.0f;
		DrawText(left, y, "Compute      CUDA / GPU");
		y += 18.0f;

		std::ostringstream grid;
		grid << "Grid         " << gApp.definition->nx
			<< " x " << gApp.definition->ny;
		DrawText(left, y, grid.str());
		y += 18.0f;

		std::ostringstream progress;
		progress << "Iteration    " << gApp.iteration;
		DrawText(left, y, progress.str());
		y += 18.0f;

		std::ostringstream speed;
		speed << "Steps/frame  " << gApp.stepsPerFrame;
		DrawText(left, y, speed.str());
		y += 18.0f;

		std::ostringstream reynolds;
		reynolds << "Re           " << std::fixed << std::setprecision(1)
			<< GetDemoCaseReynoldsNumber(*gApp.definition);
		DrawText(left, y, reynolds.str());
		y += 25.0f;

		glColor3f(0.48f, 0.72f, 0.92f);
		DrawText(left, y, "FIELD");
		y += 20.0f;
		glColor3f(0.82f, 0.84f, 0.86f);
		DrawText(left, y, GetDemoFieldViewName(gApp.fieldView));
		DrawLegend(panel, y + 8.0f);
		y += 62.0f;

		if (gApp.definition->hasMovableObstacle)
		{
			glColor3f(0.48f, 0.72f, 0.92f);
			DrawText(left, y, "PRESCRIBED SOLID");
			y += 20.0f;
			glColor3f(0.82f, 0.84f, 0.86f);
			std::ostringstream position;
			position << std::fixed << std::setprecision(1)
				<< "Center       " << gApp.obstacleX
				<< ", " << gApp.obstacleY;
			DrawText(left, y, position.str());
			y += 18.0f;

			std::ostringstream target;
			target << std::fixed << std::setprecision(1)
				<< "Target       " << gApp.obstacleTargetX
				<< ", " << gApp.obstacleTargetY;
			DrawText(left, y, target.str());
			y += 18.0f;

			std::ostringstream wallVelocity;
			wallVelocity << std::fixed << std::setprecision(3)
				<< "Wall u       " << gApp.obstacleVelocityX
				<< ", " << gApp.obstacleVelocityY;
			DrawText(left, y, wallVelocity.str());
			y += 18.0f;

			std::ostringstream load;
			load << std::showpos << std::fixed << std::setprecision(3)
				<< "Lift/drag    " << gApp.solidForceX
				<< ", " << gApp.solidForceY;
			DrawText(left, y, load.str());
			y += 25.0f;
		}
		else
		{
			glColor3f(0.48f, 0.72f, 0.92f);
			DrawText(left, y, "JET");
			y += 20.0f;
			glColor3f(0.82f, 0.84f, 0.86f);
			std::ostringstream jet;
			jet << "Nozzle       " << gApp.definition->jetWidth
				<< " cells";
			DrawText(left, y, jet.str());
			y += 18.0f;
			std::ostringstream inlet;
			inlet << std::fixed << std::setprecision(2)
				<< "Inlet u      " << gApp.definition->inletUx
				<< ", " << gApp.definition->inletUy;
			DrawText(left, y, inlet.str());
			y += 25.0f;
		}

		glColor3f(0.48f, 0.72f, 0.92f);
		DrawText(left, y, "CONTROLS");
		y += 20.0f;
		glColor3f(0.76f, 0.79f, 0.82f);
		DrawText(left, y, "1 / 2     Karman / Jet");
		y += 18.0f;
		DrawText(left, y, "V         Change field");
		y += 18.0f;
		if (gApp.definition->hasMovableObstacle)
		{
			DrawText(left, y, "Arrows    Move target");
			y += 18.0f;
		}
		DrawText(left, y, "Space     Pause / resume");
		y += 18.0f;
		DrawText(left, y, "S         Advance one frame");
		y += 18.0f;
		DrawText(left, y, "+ / -     Steps/frame [1, 50]");
		y += 18.0f;
		DrawText(left, y, "R         Reset current case");
		y += 18.0f;
		DrawText(left, y, "Esc       Quit");
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

		glViewport(0, 0, width, height);
		glClearColor(0.035f, 0.043f, 0.051f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		BeginScreenCoordinates(width, height);

		const UiLayout layout = ComputeLayout(width, height);
		DrawField(layout.field);
		DrawObstacleTarget(layout.field);
		DrawPanel(layout.panel);
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
		gApp.AdvanceObstacle();
		gApp.Step();
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

		case WM_KEYDOWN:
		{
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
			if (wParam == VK_LEFT ||
				wParam == VK_RIGHT ||
				wParam == VK_UP ||
				wParam == VK_DOWN)
			{
				gApp.SetMoveKey(wParam, false);
			}
			return 0;

		case WM_KILLFOCUS:
			gApp.ClearMoveKeys();
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

	MSG message;
	while (GetMessageW(&message, NULL, 0, 0) > 0)
	{
		TranslateMessage(&message);
		DispatchMessageW(&message);
	}
	return (int)message.wParam;
}
