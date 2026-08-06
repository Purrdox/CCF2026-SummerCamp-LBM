#pragma once

#include "LbmCases2D.h"

// §4.3 参数文件读写（与 LbmApp 零耦合，只依赖 DemoCaseDefinition / RigidBody）。
//
// 文件格式：`key=value` 每行一项，`#` 开头为注释。
// 覆盖字段：case / fieldView / stepsPerFrame / obstacleShape /
//          bodyCount / bodyN{X,Y,Radius} / viscosity / initialUx / initialUy /
//          inletUx / inletUy / inletPerturbationAmplitude /
//          inletPerturbationPeriod / speedColorMax / vorticityColorMax / jetWidth

// 返回 false 表示文件不存在或写入失败。
bool SaveAppParams(const char* path,
                   const DemoCaseDefinition& def,
                   DemoFieldView view,
                   int stepsPerFrame,
                   ObstacleShape shape,
                   int bodyCount,
                   const RigidBody* bodies /* kMaxBodies */);

// 读取参数文件。以 caseId 入参对应的默认 def 为基底再逐项覆盖，保证任何缺失
// 字段都有合法值；非法数值回退默认，不整体崩溃。
//
// 优先级约定（§4.3，写死）：命令行显式给出 --case 时，main() 传
// applyFileCase = false —— ini 不覆盖 case（只覆盖其余参数），否则文件中的
// case 键生效并覆盖 caseId 入参。
//
// 加载后的 bodies 已二次校验（§4.3）：坐标 clamp 到 [margin, nx-margin-1]
// （margin = radius+2）、半径上限 min(20, (min(nx,ny)-2)/2)、两两不重叠
// （重叠的物体丢弃）、bodyCount 截断到 [0, kMaxBodies]、caseId 非法回退默认。
// 返回 false 表示文件不存在（打开失败），不表示解析失败。
bool LoadAppParams(const char* path,
                   DemoCaseId& caseId,
                   DemoCaseDefinition& def,
                   DemoFieldView& view,
                   int& stepsPerFrame,
                   ObstacleShape& shape,
                   int& bodyCount,
                   RigidBody* bodies /* kMaxBodies */,
                   bool applyFileCase = true);
