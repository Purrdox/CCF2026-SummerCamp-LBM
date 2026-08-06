#include "LbmCases2D.h"
#include "../inc/2D/cpu/mrConstantParamsCpu2D.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace
{
	const REAL kPi = 3.14159265358979323846f;

	const DemoCaseDefinition kCases[] =
	{
		{
			DemoCaseId::KarmanVortex,
			"karman",
			"Karman vortex",
			"Uniform inlet past a circular cylinder",
			96,
			192,
			0.008f,
			0.0f,
			0.10f,
			0.0f,
			0.10f,
			0.004f,
			320,
			15,
			0.14f,
			0.050f,
			1.0f,
			DemoFieldView::Vorticity,
			true,
			47.5f,
			45.5f,
			8.0f,
			0.80f,//mark 请求 5：提高物体拖动/移动速度（原 0.28）
			0.85f,
			0
		},
		{
			DemoCaseId::JetFlow,
			"jetflow",
			"Planar jet",
			"Bottom slot jet into a quiescent channel",
			96,//mark 96
			192,//mark 192
			0.020f,//mark 粘度 0.020
			0.0f,
			0.0f,
			0.0f,
			0.08f,//mark 速度 0.08
			0.0f,
			0,
			15,
			0.10f,
			0.030f,
			1.0f,
			DemoFieldView::VelocityMagnitude,
			false,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			16
		}
	};

	std::string Lowercase(std::string value)
	{
		std::transform(
			value.begin(),
			value.end(),
			value.begin(),
			[](unsigned char c) { return (char)std::tolower(c); });
		return value;
	}

	bool IsJetInlet(const DemoCaseDefinition& definition, int x)
	{
		// ===== ASSIGNMENT FILL BEGIN: P3-A jet-slot geometry =====
		// TODO: Return true exactly on the centered bottom inlet slot.
		// The slot width is "definition.jetWidth" on a grid of "definition.nx".
		const int start = (definition.nx - definition.jetWidth) / 2;
		const int end = start + definition.jetWidth;
		return (x >= start && x < end);
		// ===== ASSIGNMENT FILL END: P3-A =====
	}

	bool IsFiniteMoment(REAL value)
	{
		return std::isfinite((double)value) != 0;
	}

	void WriteBoundaryMoments(
		mrFlow2D* flow,
		int boundaryIndex,
		int neighborIndex,
		REAL targetUx,
		REAL targetUy)
	{
		REAL rho = flow->fMom[neighborIndex * 6 + 0];
		const REAL neighborUx = flow->fMom[neighborIndex * 6 + 1];
		const REAL neighborUy = flow->fMom[neighborIndex * 6 + 2];
		const REAL neighborSxx = flow->fMom[neighborIndex * 6 + 3];
		const REAL neighborSyy = flow->fMom[neighborIndex * 6 + 4];
		const REAL neighborSxy = flow->fMom[neighborIndex * 6 + 5];

		if (!IsFiniteMoment(rho) || rho < 0.5f || rho > 1.5f)
		{
			rho = 1.0f;
		}

		if (!IsFiniteMoment(neighborUx) ||
			!IsFiniteMoment(neighborUy) ||
			!IsFiniteMoment(neighborSxx) ||
			!IsFiniteMoment(neighborSyy) ||
			!IsFiniteMoment(neighborSxy))
		{
			WriteEquilibriumMoments2D(
				flow, boundaryIndex, rho, targetUx, targetUy);
			return;
		}

		// Non-equilibrium extrapolation: change the prescribed velocity while
		// retaining the stress carried by the adjacent fluid cell.
		const REAL sxx = targetUx * targetUx +
			(neighborSxx - neighborUx * neighborUx);
		const REAL syy = targetUy * targetUy +
			(neighborSyy - neighborUy * neighborUy);
		const REAL sxy = targetUx * targetUy +
			(neighborSxy - neighborUx * neighborUy);
		WriteMoments2D(
			flow, boundaryIndex, rho, targetUx, targetUy, sxx, syy, sxy);
	}
}

DemoCaseDefinition GetDefaultDefinition(DemoCaseId id)
{
	for (const DemoCaseDefinition& definition : kCases)
	{
		if (definition.id == id)
		{
			return definition;
		}
	}
	return kCases[0];
}

bool ParseDemoCaseName(const std::string& name, DemoCaseId& id)
{
	const std::string normalized = Lowercase(name);
	if (normalized == "karman" || normalized == "karman-vortex" || normalized == "vortex")
	{
		id = DemoCaseId::KarmanVortex;
		return true;
	}
	if (normalized == "jet" || normalized == "jetflow" || normalized == "jet-flow")
	{
		id = DemoCaseId::JetFlow;
		return true;
	}
	return false;
}

float GetDemoCaseReynoldsNumber(const DemoCaseDefinition& definition)
{
	const float characteristicLength = definition.hasMovableObstacle
		? 2.0f * definition.obstacleRadius
		: (float)definition.jetWidth;
	const float inletSpeed = sqrtf(
		definition.inletUx * definition.inletUx +
		definition.inletUy * definition.inletUy);
	return inletSpeed * characteristicLength / definition.viscosity;
}

const char* GetDemoFieldViewName(DemoFieldView view)
{
	if (view == DemoFieldView::Vorticity)
	{
		return "signed vorticity";
	}
	if (view == DemoFieldView::Colorful)
	{
		return "colorful flow";
	}
	return "velocity magnitude";
}

MLLATTICENODE_FLAG GetDemoCaseBaseFlag(
	const DemoCaseDefinition& definition,
	int x,
	int y)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-P3-A case boundary flags =====
	// TODO: Classify the fixed domain boundary for both cases.
	// Karman: full bottom inlet, top/left/right open outlets.
	// Jet: centered bottom inlet, remaining bottom and both sides no-slip
	// walls, and an open top outlet. Interior cells are fluid.
	if (definition.id == DemoCaseId::KarmanVortex)
	{
		if (y == 0) return ML_INLET;
		if (x == 0 || x == definition.nx - 1 || y == definition.ny - 1)
		{
			return ML_OUTLET;
		}
		return ML_FLUID;
	}

	if (definition.id == DemoCaseId::JetFlow)
	{
		if (y == 0)
		{
			return IsJetInlet(definition, x) ? ML_INLET : ML_WALL_DOWN;
		}
		if (x == 0) return ML_WALL_LEFT;
		if (x == definition.nx - 1) return ML_WALL_RIGHT;
		if (y == definition.ny - 1) return ML_OUTLET;
		return ML_FLUID;
	}

	return ML_FLUID;
	// ===== ASSIGNMENT FILL END: P2-P3-A =====
}

bool IsObstacleCell(
	ObstacleShape shape,
	int x,
	int y,
	float cx,
	float cy,
	float radius)
{
	const float dx = (float)x - cx;
	const float dy = (float)y - cy;

	if (shape == ObstacleShape::Circle)
	{
		const float dist2 = dx * dx + dy * dy;
		const float r2 = radius * radius;
		return dist2 <= r2;
	}

	if (shape == ObstacleShape::Box)
	{
		// Box 为半宽 = 半高：以中心为中心的方形
		return fabsf(dx) <= radius && fabsf(dy) <= radius;
	}

	// Diamond：对角线长度为 2 * radius 的旋转 45° 方形
	return fabsf(dx) + fabsf(dy) <= radius;
}

bool IsAnyObstacleCell(
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape,
	int x,
	int y)
{
	for (int i = 0; i < bodyCount; i++)
	{
		if (IsObstacleCell(shape, x, y, bodies[i].x, bodies[i].y, bodies[i].radius))
		{
			return true;
		}
	}
	return false;
}

int OwnerBodyOfCell(
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape,
	int x,
	int y)
{
	// 从后往前遍历（后添加者优先），与 PickBody 一致
	for (int i = bodyCount - 1; i >= 0; i--)
	{
		if (IsObstacleCell(shape, x, y, bodies[i].x, bodies[i].y, bodies[i].radius))
		{
			return i;
		}
	}
	return -1;
}

const char* GetObstacleShapeName(ObstacleShape shape)
{
	switch (shape)
	{
	case ObstacleShape::Circle: return "Circle";
	case ObstacleShape::Box: return "Box";
	case ObstacleShape::Diamond: return "Diamond";
	}
	return "Circle";
}

void WriteMoments2D(
	mrFlow2D* flow,
	int index,
	REAL rho,
	REAL ux,
	REAL uy,
	REAL sxx,
	REAL syy,
	REAL sxy)
{
	flow->fMomPost[index * 6 + 0] = flow->fMom[index * 6 + 0] = rho;
	flow->fMomPost[index * 6 + 1] = flow->fMom[index * 6 + 1] = ux;
	flow->fMomPost[index * 6 + 2] = flow->fMom[index * 6 + 2] = uy;
	flow->fMomPost[index * 6 + 3] = flow->fMom[index * 6 + 3] = sxx;
	flow->fMomPost[index * 6 + 4] = flow->fMom[index * 6 + 4] = syy;
	flow->fMomPost[index * 6 + 5] = flow->fMom[index * 6 + 5] = sxy;
	flow->forcex[index] = 0.0f;
	flow->forcey[index] = 0.0f;
}

void WriteEquilibriumMoments2D(
	mrFlow2D* flow,
	int index,
	REAL rho,
	REAL ux,
	REAL uy)
{
	REAL population[9];
	const REAL u2 = ux * ux + uy * uy;

	for (int i = 0; i < 9; i++)
	{
		const REAL cu = ex2d_cpu[i] * ux + ey2d_cpu[i] * uy;
		population[i] = w2d_cpu[i] * rho *
			(1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
	}

	const REAL invRho = 1.0f / rho;
	REAL pixx = population[1] + population[3] + population[5] +
		population[6] + population[7] + population[8];
	REAL piyy = population[2] + population[4] + population[5] +
		population[6] + population[7] + population[8];
	REAL pixy = population[5] - population[6] + population[7] - population[8];
	pixx = pixx * invRho - cs2_cpu;
	piyy = piyy * invRho - cs2_cpu;
	pixy *= invRho;

	WriteMoments2D(flow, index, rho, ux, uy, pixx, piyy, pixy);
}

void InitializeDemoCase(
	mrFlow2D* flow,
	const DemoCaseDefinition& definition,
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-P3-B case initialization =====
	// TODO: Initialize every lattice cell for the selected case.
	// Combine the fixed boundary flag with the optional obstacle bodies,
	// choose initial/inlet/no-slip velocity, and initialize both moment
	// buffers with equilibrium moments at rho = 1.
	for (int y = 0; y < definition.ny; y++)
	{
		for (int x = 0; x < definition.nx; x++)
		{
			const int idx = y * definition.nx + x;
			MLLATTICENODE_FLAG flag = GetDemoCaseBaseFlag(definition, x, y);
			const bool solid = IsAnyObstacleCell(bodies, bodyCount, shape, x, y);
			if (solid)
			{
				flag = ML_SOLID;
			}
			flow->flag[idx] = flag;

			REAL ux = 0.0f, uy = 0.0f;
			if (flag == ML_INLET)
			{
				ux = definition.inletUx;
				uy = definition.inletUy;
			}
			else if (flag == ML_FLUID)
			{
				ux = definition.initialUx;
				uy = definition.initialUy;
			}

			WriteEquilibriumMoments2D(flow, idx, 1.0f, ux, uy);
		}
	}
	// ===== ASSIGNMENT FILL END: P2-P3-B =====
}

void RefreshDemoCaseBoundaries(
	mrFlow2D* flow,
	const DemoCaseDefinition& definition,
	int iteration)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-P3-C time-dependent boundaries =====
	// TODO: Refresh all outer-boundary moments before each UI frame.
	// Add the configured transverse perturbation to the Karman inlet, find
	// each boundary cell's adjacent interior cell, prescribe inlet velocity,
	// copy interior velocity at open outlets, and use zero velocity at walls.
	// Call WriteBoundaryMoments so non-equilibrium stress is extrapolated.
	const REAL perturbation = (definition.id == DemoCaseId::KarmanVortex)
		? definition.inletPerturbationAmplitude *
			sinf(2.0f * kPi * (REAL)iteration / (REAL)definition.inletPerturbationPeriod)
		: 0.0f;

	// Bottom boundary (y = 0): inlet or bottom wall, including the corners.
	for (int x = 0; x < definition.nx; x++)
	{
		const int idx = 0 * definition.nx + x;
		const int neighborIdx = 1 * definition.nx + x;
		REAL ux = 0.0f, uy = 0.0f;
		if (flow->flag[idx] == ML_INLET)
		{
			ux = definition.inletUx + perturbation;
			uy = definition.inletUy;
		}
		WriteBoundaryMoments(flow, idx, neighborIdx, ux, uy);
	}

	// Top boundary (y = ny - 1): open outlet. Use a convective (Orlanski)
	// outlet that extrapolates the velocity toward the boundary from the two
	// interior rows, letting vorticity leave the domain without reflecting
	// perturbations back (important at high Reynolds numbers).
	for (int x = 0; x < definition.nx; x++)
	{
		const int idx = (definition.ny - 1) * definition.nx + x;
		const int neighborIdx = (definition.ny - 2) * definition.nx + x;
		const int secondIdx = (definition.ny - 3) * definition.nx + x;
		const REAL ux = flow->fMom[neighborIdx * 6 + 1] -
			0.9f * (flow->fMom[neighborIdx * 6 + 1] - flow->fMom[secondIdx * 6 + 1]);
		const REAL uy = flow->fMom[neighborIdx * 6 + 2] -
			0.9f * (flow->fMom[neighborIdx * 6 + 2] - flow->fMom[secondIdx * 6 + 2]);
		WriteBoundaryMoments(flow, idx, neighborIdx, ux, uy);
	}

	// Left boundary (x = 0): wall for jet, outlet for Karman. Corners are
	// already covered by the bottom/top loops above.
	for (int y = 1; y < definition.ny - 1; y++)
	{
		const int idx = y * definition.nx + 0;
		const int neighborIdx = y * definition.nx + 1;
		REAL ux = 0.0f, uy = 0.0f;
		if (flow->flag[idx] == ML_OUTLET)
		{
			ux = flow->fMom[neighborIdx * 6 + 1];
			uy = flow->fMom[neighborIdx * 6 + 2];
		}
		WriteBoundaryMoments(flow, idx, neighborIdx, ux, uy);
	}

	// Right boundary (x = nx - 1): wall for jet, outlet for Karman.
	for (int y = 1; y < definition.ny - 1; y++)
	{
		const int idx = y * definition.nx + (definition.nx - 1);
		const int neighborIdx = y * definition.nx + (definition.nx - 2);
		REAL ux = 0.0f, uy = 0.0f;
		if (flow->flag[idx] == ML_OUTLET)
		{
			ux = flow->fMom[neighborIdx * 6 + 1];
			uy = flow->fMom[neighborIdx * 6 + 2];
		}
		WriteBoundaryMoments(flow, idx, neighborIdx, ux, uy);
	}
	// ===== ASSIGNMENT FILL END: P2-P3-C =====
}
