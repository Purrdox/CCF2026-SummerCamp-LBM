#pragma once

#include "../inc/2D/cpu/mrFlow2D.h"

#include <string>

enum class DemoCaseId
{
	KarmanVortex,
	JetFlow
};

enum class DemoFieldView
{
	VelocityMagnitude,
	Vorticity
};

struct DemoCaseDefinition
{
	DemoCaseId id;
	const char* cliName;
	const char* displayName;
	const char* description;
	int nx;
	int ny;
	REAL viscosity;
	REAL initialUx;
	REAL initialUy;
	REAL inletUx;
	REAL inletUy;
	REAL inletPerturbationAmplitude;
	int inletPerturbationPeriod;
	int initialStepsPerFrame;
	float speedColorMax;
	float vorticityColorMax;
	DemoFieldView defaultView;
	bool hasMovableObstacle;
	float obstacleStartX;
	float obstacleStartY;
	float obstacleRadius;
	float obstacleMoveSpeed;
	float obstacleTargetSpeed;
	int jetWidth;
};

const DemoCaseDefinition& GetDemoCaseDefinition(DemoCaseId id);
bool ParseDemoCaseName(const std::string& name, DemoCaseId& id);
float GetDemoCaseReynoldsNumber(const DemoCaseDefinition& definition);
const char* GetDemoFieldViewName(DemoFieldView view);

MLLATTICENODE_FLAG GetDemoCaseBaseFlag(
	const DemoCaseDefinition& definition,
	int x,
	int y);

bool IsDemoCaseObstacleCell(
	const DemoCaseDefinition& definition,
	int x,
	int y,
	float obstacleX,
	float obstacleY);

void WriteMoments2D(
	mrFlow2D* flow,
	int index,
	REAL rho,
	REAL ux,
	REAL uy,
	REAL sxx,
	REAL syy,
	REAL sxy);

void WriteEquilibriumMoments2D(
	mrFlow2D* flow,
	int index,
	REAL rho,
	REAL ux,
	REAL uy);

void InitializeDemoCase(
	mrFlow2D* flow,
	const DemoCaseDefinition& definition,
	float obstacleX,
	float obstacleY);

void RefreshDemoCaseBoundaries(
	mrFlow2D* flow,
	const DemoCaseDefinition& definition,
	int iteration);
