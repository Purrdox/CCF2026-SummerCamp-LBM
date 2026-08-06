#pragma once

#include "../inc/2D/cpu/mrFlow2D.h"

#include <cstdint>
#include <string>

// 全局单一形状，所有物体共用（组合 A 约束：不旋转、不重叠、上限 kMaxBodies 个）
enum class ObstacleShape
{
	Circle,
	Box,
	Diamond
};

const int kMaxBodies = 4;

struct RigidBody
{
	float x, y;      // 当前中心位置（格点坐标）
	float vx, vy;    // 壁面速度（每步 = 位移 / stepsPerFrame）
	float tx, ty;    // 目标点（键盘 / 鼠标 / UI 写入）
	float radius;    // 半径；Box 为半宽 = 半高
	bool selected;   // 选中态（拖动高亮 + 面板显示）
};

enum class DemoCaseId
{
	KarmanVortex,
	JetFlow
};

enum class DemoFieldView
{
	VelocityMagnitude,
	Vorticity,
	Colorful,   // 功能 4：rho/ux/uy 流场着色（平静=白，快速=鲜艳+时间旋转）
	Smoke       // 烟雾视图：仅显示烟雾状态（被动标量颜色场），不含任何常规流体信息
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
	float colorfulSaturation;   // 功能 4：colorful 视图颜色鲜艳度（0=全白，1=全鲜艳，随时间/方向渲染）
	DemoFieldView defaultView;
	bool hasMovableObstacle;
	float obstacleStartX;
	float obstacleStartY;
	float obstacleRadius;
	float obstacleMoveSpeed;
	float obstacleTargetSpeed;
	int jetWidth;
};

// 运行时预设（功能 3，§4.2）
struct CasePreset
{
	const char* name;   // "Karman", "Jet", "Custom"
	DemoCaseId id;
	DemoCaseDefinition def;  // 默认参数快照
};

// 返回 kCases[] 中对应 id 定义的拷贝（kCases 保持 const，作为"出厂默认值"来源）
DemoCaseDefinition GetDefaultDefinition(DemoCaseId id);
bool ParseDemoCaseName(const std::string& name, DemoCaseId& id);
float GetDemoCaseReynoldsNumber(const DemoCaseDefinition& definition);
const char* GetDemoFieldViewName(DemoFieldView view);

MLLATTICENODE_FLAG GetDemoCaseBaseFlag(
	const DemoCaseDefinition& definition,
	int x,
	int y);

// 格点判据（§1.2）：互不重叠 ⇒ OwnerBodyOfCell 命中唯一，未命中返回 -1
bool IsObstacleCell(
	ObstacleShape shape,
	int x,
	int y,
	float cx,
	float cy,
	float radius);

bool IsAnyObstacleCell(
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape,
	int x,
	int y);

int OwnerBodyOfCell(
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape,
	int x,
	int y);

const char* GetObstacleShapeName(ObstacleShape shape);

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
	const RigidBody* bodies,
	int bodyCount,
	ObstacleShape shape);

void RefreshDemoCaseBoundaries(
	mrFlow2D* flow,
	const DemoCaseDefinition& definition,
	int iteration);
