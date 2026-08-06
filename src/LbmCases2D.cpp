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
			DemoFieldView::Vorticity,
			true,
			47.5f,
			45.5f,
			8.0f,
			0.28f,
			0.85f,
			0
		},
		{
			DemoCaseId::JetFlow,
			"jetflow",
			"Planar jet",
			"Bottom slot jet into a quiescent channel",
			96,
			192,
			0.020f,
			0.0f,
			0.0f,
			0.0f,
			0.08f,
			0.0f,
			0,
			15,
			0.10f,
			0.030f,
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
		// The slot is centered at the bottom, covering exactly
		// definition.jetWidth cells.
		int center = definition.nx / 2;
		int start = center - definition.jetWidth / 2;
		int end = start + definition.jetWidth;
		return x >= start && x < end;
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

const DemoCaseDefinition& GetDemoCaseDefinition(DemoCaseId id)
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
	return view == DemoFieldView::Vorticity
		? "signed vorticity"
		: "velocity magnitude";
}

MLLATTICENODE_FLAG GetDemoCaseBaseFlag(
	const DemoCaseDefinition& definition,
	int x,
	int y)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-P3-A case boundary flags =====
	// Classify the fixed domain boundary for both Karman and Jet cases.

	bool onLeft   = (x == 0);
	bool onRight  = (x == definition.nx - 1);
	bool onBottom = (y == 0);
	bool onTop    = (y == definition.ny - 1);

	if (definition.id == DemoCaseId::KarmanVortex)
	{
		// Karman: bottom inlet, top / left / right open outlets.
		if (onBottom)
			return ML_INLET;
		if (onTop || onLeft || onRight)
			return ML_OUTLET;
		return ML_FLUID;
	}
	else // JetFlow
	{
		// Jet: centered bottom inlet, remaining bottom + both sides walls,
		// top open outlet.
		if (onBottom)
			return IsJetInlet(definition, x) ? ML_INLET : ML_WALL_DOWN;
		if (onTop)
			return ML_OUTLET;
		if (onLeft)
			return ML_WALL_LEFT;
		if (onRight)
			return ML_WALL_RIGHT;
		return ML_FLUID;
	}
	// ===== ASSIGNMENT FILL END: P2-P3-A =====
}

bool IsDemoCaseObstacleCell(
	const DemoCaseDefinition& definition,
	int x,
	int y,
	float obstacleX,
	float obstacleY)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-A Karman cylinder geometry =====
	// Identify lattice cells covered by the movable circular obstacle.
	// Cases without an obstacle must never report a solid cell.
	if (!definition.hasMovableObstacle)
		return false;

	REAL dx = (REAL)x - obstacleX;
	REAL dy = (REAL)y - obstacleY;
	REAL dist2 = dx * dx + dy * dy;
	REAL r = definition.obstacleRadius;
	return dist2 <= r * r;
	// ===== ASSIGNMENT FILL END: P2-A =====
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
	float obstacleX,
	float obstacleY)
{
	// ===== ASSIGNMENT FILL BEGIN: P2-P3-B case initialization =====
	// Initialize every lattice cell for the selected case.
	int nx = definition.nx;
	int ny = definition.ny;

	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			int index = y * nx + x;

			// Combine fixed boundary flag with optional Karman cylinder.
			MLLATTICENODE_FLAG flag = GetDemoCaseBaseFlag(definition, x, y);
			if (definition.hasMovableObstacle &&
				IsDemoCaseObstacleCell(definition, x, y, obstacleX, obstacleY))
			{
				flag = ML_SOLID;
			}
			flow->flag[index] = flag;

			// Choose initial/inlet/no-slip velocity.
			REAL ux = definition.initialUx;
			REAL uy = definition.initialUy;

			if (flag == ML_INLET)
			{
				ux = definition.inletUx;
				uy = definition.inletUy;
			}
			else if (flag == ML_WALL || flag == ML_WALL_LEFT ||
				flag == ML_WALL_RIGHT || flag == ML_WALL_DOWN ||
				flag == ML_WALL_UP || flag == ML_SOLID)
			{
				ux = 0.0f;
				uy = 0.0f;
			}

			// Initialize both moment buffers with equilibrium at rho = 1.
			WriteEquilibriumMoments2D(flow, index, 1.0f, ux, uy);
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
	// Refresh all outer-boundary moments before each UI frame.
	int nx = definition.nx;
	int ny = definition.ny;
	int perturbationSign = sinf(2.0f * kPi * (REAL)iteration /
		(REAL)definition.inletPerturbationPeriod);

	for (int x = 0; x < nx; x++)
	{
		for (int y = 0; y < ny; y++)
		{
			// Only process boundary cells.
			if (!(x == 0 || x == nx - 1 || y == 0 || y == ny - 1))
				continue;

			int index = y * nx + x;
			MLLATTICENODE_FLAG flag = flow->flag[index];
			if (flag == ML_SOLID)
				continue;

			// Determine neighbor index (one cell inward from the boundary).
			int nxCell = x;
			int nyCell = y;
			if (y == 0)
				nyCell = 1;
			else if (y == ny - 1)
				nyCell = ny - 2;
			if (x == 0)
				nxCell = 1;
			else if (x == nx - 1)
				nxCell = nx - 2;
			int neighborIndex = nyCell * nx + nxCell;

			if (flag == ML_INLET)
			{
				// Inlet velocity.
				REAL ux = definition.inletUx;
				REAL uy = definition.inletUy;

				// Karman inlet perturbation in the transverse direction.
				if (definition.id == DemoCaseId::KarmanVortex)
				{
					ux += definition.inletPerturbationAmplitude *
						(REAL)perturbationSign;
				}

				WriteBoundaryMoments(flow, index, neighborIndex, ux, uy);
			}
			else if (flag == ML_OUTLET)
			{
				// Open outlet: copy interior velocity.
				REAL ux = flow->fMom[neighborIndex * 6 + 1];
				REAL uy = flow->fMom[neighborIndex * 6 + 2];
				WriteBoundaryMoments(flow, index, neighborIndex, ux, uy);
			}
			else
			{
				// Wall: zero velocity, extrapolate non-equilibrium stress.
				WriteBoundaryMoments(flow, index, neighborIndex, 0.0f, 0.0f);
			}
		}
	}
	// ===== ASSIGNMENT FILL END: P2-P3-C =====
}
