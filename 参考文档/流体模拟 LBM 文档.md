# 二维格子 Boltzmann 流体模拟实验报告

> 一个基于 **D2Q9 格子 Boltzmann 方法（LBM）** 的二维流场实时交互程序：CUDA GPU 完成流场推进，CPU 负责算例初始化、边界刷新、移动固体与显示数据准备，OpenGL 渲染画面，ImGui 提供交互面板。

---

## 一、项目概述

### 1.1 技术栈

| 层 | 技术 |
| --- | --- |
| 求解 | CUDA（`mrSolver2DKernel`，GPU 每步迭代） |
| 渲染 | OpenGL（固定管线画像素纹理） |
| 界面 | Dear ImGui（1.90.9，Win32 + OpenGL2 后端） |
| 语言 | C++ / MSVC v143（`Release|x64`） |

### 1.2 目录结构

```text
Home2d/
├── LBM_MRLBM.sln / .vcxproj      # 工程
├── 3rdParty/                     # imgui、CUDA 辅助头文件
├── common/                       # 公共类型、颜色映射、CUDA 包装（mlCuRunTime 等）
├── inc/2D/
│   ├── cpu/                      # mrFlow2D / mrSolver2D / mrConstantParamsCpu2D / mlInit2D
│   └── gpu/                      # mrConstantParamsGpu2D / mrUtilFuncGpu2D / mrLbmSolverGpu2D(.cu)
├── src/
│   ├── LbmCases2D.h/.cpp         # 算例几何、边界标记、初始化、边界刷新
│   ├── ParamsIO.h/.cpp           # 参数文件（ini）读写
│   └── testMrLBM2D.cpp           # 程序入口 + 全部交互/显示/烟雾/存档功能
└── x64/Release/LBM_MRLBM.exe     # 可执行文件
```

---

## 二、编译与运行

1. 用 Visual Studio 2022 打开 `LBM_MRLBM.sln`，选择 `Release | x64` 生成；或命令行：

   ```powershell
   msbuild LBM_MRLBM.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
   ```

2. 运行：

   ```powershell
   .\x64\Release\LBM_MRLBM.exe            # 默认 Karman 算例
   .\x64\Release\LBM_MRLBM.exe --case jetflow
   .\x64\Release\LBM_MRLBM.exe --help
   ```

   > 启动时会自动读取**可执行目录**下的 `params.ini`（不存在则忽略）；命令行显式 `--case` 时 ini 不覆盖 case。

---

## 三、操作指南

### 3.1 键盘快捷键

| 按键 | 功能 |
| --- | --- |
| `1` / `2` | 重建并切换算例：Karman 涡街 / Jet 射流 |
| `V` | 循环切换基本视图：速度 → 涡量 → 烟雾（Colorful 需从面板手动进入） |
| `Space` | 暂停 / 继续 |
| `S` | 暂停时手动推进一个 UI 帧 |
| `+` / `-` | 调整每帧 LBM 迭代步数 steps/frame |
| `R`（短按） | Restart：**按当前已调参数**重建算例 |
| `R`（长按 ≥500ms） | Reset：恢复该算例默认设置 |
| `Tab` | 循环选中物体 |
| `↑↓←→` | 移动选中物体 |
| `Ctrl` | 按住进入调试模式（鼠标处放大镜 + Ctrl+左键看格点数据） |
| `Esc` | 退出 |

### 3.2 鼠标操作

| 操作 | 功能 |
| --- | --- |
| 左键（选中工具时） | 按当前工具作用：烟雾画笔 / 擦除 / 吹风 |
| Shift + 左键 | 吹风（Blow，恒可用，不依赖工具选择） |
| 中键 | 旋涡（Vortex，恒可用） |
| 左键拖动物体 | 拖动/移动选中物体 |
| 鼠标悬停（选中工具） | 显示工具实际作用范围（浅青色椭圆） |

### 3.3 右侧控制面板（ImGui）

| 分组 | 内容 |
| --- | --- |
| UI size | 小 / 中 / 大三种窗口尺寸模式 |
| SIMULATION | 运行状态、steps/frame 滑块、Pause/Step/Reset/Restart、Field view 视图下拉 |
| CASE / PRESET | Case 切换、预设下拉（Karman/Jet/Blank）、Reynolds 数、Restore defaults、Save ini / Load ini |
| PARAMETERS | 粘度、入口速度 ux/uy、入口扰动幅值/周期、按视图显示的颜色上限滑块 |
| OBJECTS | 形状（Circle/Box/Diamond）、Add body、物体列表、Center X/Y、Radius、Remove selected、受力信息 |
| Tool | 工具按钮（Blow/Vortex/Smoke/Eraser）、烟雾调色板、Brush strength、Smoke temperature、Brush radius、Blow strength |
| DEBUG | Show FPS overlay 开关、Ctrl 调试说明 |
| CONTROLS | 快捷键速查 |

---

## 四、LBM 数值方法基础

### 4.1 D2Q9 离散速度模型

粒子在每个格点沿 9 个方向运动：

| `i` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `(ex, ey)` | (0,0) | (1,0) | (0,1) | (-1,0) | (0,-1) | (1,1) | (-1,1) | (-1,-1) | (1,-1) |
| 权重 `w` | 4/9 | 1/9 | 1/9 | 1/9 | 1/9 | 1/36 | 1/36 | 1/36 | 1/36 |

声速平方 `cs² = 1/3`。方向与权重常量定义在 `inc/2D/cpu/mrConstantParamsCpu2D.h` 与 `inc/2D/gpu/mrConstantParamsGpu2D.h`。

### 4.2 六矩存储（项目特色）

不长期存 9 个分布函数 `f_i`，而存 6 个矩（`fMom` / `fMomPost` 双缓冲，`index*6` 偏移）：

| 偏移 | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 内容 | 密度 ρ | ux | uy | 归一化二阶矩 pxx | pyy | pxy |

分布函数按需从矩重构（P1-A），一次迭代后交换两个缓冲指针（`MomSwap`）。

### 4.3 单步算法（streaming + collision）

对每个流体格点：

1. **Pull streaming**：沿方向 `i` 拉取上游邻居 `(x-ex[i], y-ey[i])` 的矩，重构出 `f_i`；
2. **宏观量恢复**：`ρ = Σf_i`，`ρu = Σf_i·e_i`（含 Guo 体力半时间步修正 `u = (Σf e + 0.5F)/ρ`）；
3. **BGK 碰撞**：松弛参数 `ω = 1/τ`，`τ = 3ν + 0.5`，二阶矩向平衡态松弛 `π ← π - ω(π - π_eq)`；
4. **归一化写回** `fMomPost`。

### 4.4 边界条件

- **入口 INLET**：规定速度（Karman 加横向正弦扰动）；**出口 OUTLET**：复制相邻内部格点速度（顶部用二阶外推）；**固壁 WALL**：零速度。统一由 `WriteBoundaryMoments` 写矩——保留邻居的非平衡应力、只替换速度（非平衡外推）。
- 边界格点矩由 CPU 每帧预先填好，GPU pull streaming 自然把边界信息带进流体——这是边界条件向内部传播的关键。

### 4.5 体力

吹风/旋涡等工具把力写入 `forcex/forcey`，在宏观量恢复与碰撞中用 **Guo 格式**（半时间步 + 二阶矩修正）计入。

---

## 五、核心求解器实现（基础部分）

### 5.1 P1-A 分布函数重构 `mlCalDistributionD2Q9AtIndex`

**文件**：`inc/2D/gpu/mrUtilFuncGpu2D.h`

由 ρ、u、归一化二阶矩重构方向 `i` 的分布函数：先用 Hermite 展开计算原始矩系数 `a0/ax/ay/axx/ayy/axy`（以及递推的三阶系数 axxy/axyy），再代入各方向的 D2Q9 多项式，乘权重 `w_i`，最后做非负截断。

### 5.2 P1-B BGK 碰撞 `mlGetPIAfterCollision`

**文件**：`inc/2D/gpu/mrUtilFuncGpu2D.h`

对三个**原始**二阶矩原地碰撞：`π_eq(ρ,u) = ρ(u·u + cs²)`，`π ← π - ω(π - π_eq)`；再叠加 Guo 体力修正项（因子 `1-0.5ω`）。

### 5.3 P1-C 求解内核 `mrSolver2DKernel`

**文件**：`inc/2D/gpu/mrLbmSolverGpu2D.cu`

线程一格点，只处理 `ML_FLUID`；依次完成 §4.3 的四个步骤，写 `fMomPost`，随后交换缓冲。CPU 侧 `mrSolver2D`（`inc/2D/cpu/mrSolver2D.h`）负责内存管理、双缓冲交换与 kernel 启动。

### 5.4 每帧数据流

```text
Init(mrFlow2D 创建 + InitializeDemoCase)
  → mlTransData2Gpu（全量上传 GPU）
  → 每 UI 帧：
      RefreshDemoCaseBoundaries（host 边界矩）
      → SyncOuterBoundaryMomentsToDevice
      → stepsPerFrame 次 mlIterateGpu（内核 + MomSwap，异步队列化）
      → CopyMomentsFromDevice（D2H 同步拷贝，隐式等待全部 kernel）
      → BuildFieldImage（生成像素图）→ OpenGL 绘制
```

### 5.5 CUDA 基础优化（本轮）

**文件**：`inc/2D/gpu/mrLbmSolverGpu2D.cu`

以"保稳定、不动物理公式"为前提做基础加速，数值与原实现逐位一致（仅改变线程→格点映射与指令调度，物理公式/边界钳制逻辑均未动）：

| # | 优化项 | 原实现 | 现实现 | 收益 |
| --- | --- | --- | --- | --- |
| 1 | 线程块映射 | 2D 块 8×8=64 线程，`y*nx+x` 索引 | 1D 块 256 线程，线程线性映射格点（`index`，再拆 x/y） | warp 内 32 个连续格点访存完全合并；块占用提升 |
| 2 | `__restrict__` | 无 | 内核指针加 `__restrict__` | 消除别名假设，利于寄存器分配与指令调度 |
| 3 | `#pragma unroll` | 9 方向循环靠编译器启发式 | 两个 9 次循环显式 `#pragma unroll` | 循环索引变常量，消掉 switch 分支，提升 ILP |
| 4 | 松弛参数提前算 | 每格重复读 `vis_shear` + 除法 | 格点入口一次计算 `tau/omega` | 消除逐方向重复访存与除法 |
| 5 | 取消每步同步 | 每步 2 次 `cudaDeviceSynchronize()` | 仅保留 `cudaGetLastError()` 错误捕获 | 消除帧内 2×stepsPerFrame 次同步延迟，kernel 异步队列化 |

**稳定性保障**：

- 优化 1-4 仅改变内存布局与指令调度，计算结果逐位一致；
- 优化 5 的正确性依赖默认流的有序性：边界 H2D 同步 → 迭代 kernel → 帧末 `CopyMomentsFromDevice`（`_MLCuMemcpy` = 同步 `cudaMemcpy`，隐式等待全部 kernel 完成）→ host 读取，时序不变；错误仍在每步 `GetLastError` 捕获；
- **暂缓项**（保稳定，后续可做）：shared memory 分块 tile（进一步削减 9 次邻居全局读）、float4 向量化读写（当前每格 6 矩 24B 步长非 16B 对齐，需先改存储布局）。

---

## 六、算例实现（`src/LbmCases2D.h/.cpp`）

| 算例 | 网格 | 参数 | 边界 |
| --- | --- | --- | --- |
| Karman 涡街 | 96×192 | ν=0.008，U=0.10，Re=200，圆柱 R=8（可移动） | 底部全宽入口，顶/左/右开放出口 |
| Jet 射流 | 96×192 | ν=0.020，U=0.08，Re=64，喷口宽 16 | 底部中央喷口 + 两侧固壁，左右固壁，顶部出口 |

涉及函数：

- `GetDemoCaseBaseFlag`（P2-P3-A）：按算例给每个格点分派边界标记（INLET/OUTLET/WALL_*/FLUID）；
- `IsObstacleCell` / `IsAnyObstacleCell` / `OwnerBodyOfCell`（P2-A 扩展）：圆/方形/菱形三种形状的格点判定；
- `InitializeDemoCase`（P2-P3-B）：叠加固体标记，按格点类型选速度，以 `ρ=1` 平衡态写双缓冲；
- `RefreshDemoCaseBoundaries`（P2-P3-C）：逐帧刷新四边边界矩，Karman 入口加横向正弦扰动（周期 320 迭代）；
- `IsJetInlet`（P3-A）：底边居中喷口判定。

---

## 七、交互应用与全部功能（`src/testMrLBM2D.cpp`）

程序主体 `LbmApp` 结构体 + `WindowProcedure` 窗口过程 + 每帧 `Step()` 执行序列：

```text
ApplyMouseEffects（工具注入力/烟） → RefreshDemoCaseBoundaries → 同步边界
→ stepsPerFrame 次 GPU 迭代 → CopyMomentsFromDevice → 清力 → AdvectSmoke
→ ComputeSolidLoads（物体受力） → BuildFieldImage → 渲染
```

### 7.1 显示系统：四种视图

| 视图 | 内容 | 实现要点 |
| --- | --- | --- |
| Velocity | 速度模长（色带） | `BuildFieldImage` 分支 |
| Vorticity | 带符号涡量 | 由邻居速度差分近似 |
| Colorful | ρ/ux/uy 综合着色，鲜艳度可调 | 鲜艳度 `colorfulSaturation` |
| Smoke | 只显示烟雾颜色（RGB 被动标量） | 直接读 `smoke[]`，不显示任何流体信息 |

文件：`src/testMrLBM2D.cpp`（`BuildFieldImage`、`ToggleFieldView`、面板 Combo、`DrawLegendOverlay` 色条）+ `src/LbmCases2D.cpp`（`GetDemoFieldViewName`）+ `src/ParamsIO.cpp`（ini 解析 `fieldView=3`）。

> **视图切换规则**：`V` 键只在 **速度 → 涡量 → 烟雾** 三视图间循环（`ToggleFieldView`），已从默认循环中移除 Colorful；Colorful 仍保留在面板 `Field view` 下拉中，可手动切换进入（从 Colorful 按 `V` 则回到速度视图）。

### 7.2 交互工具

| 工具 | 触发 | 实现 |
| --- | --- | --- |
| Blow 吹风 | Shift+左键 / 选中后左键 | 把力写入 `forcex/forcey`（方向 = 帧间鼠标差分），收集进 `injectedForceCells` 并同步设备，`ClearInjectedForces` 保证力只活一帧 |
| Vortex 旋涡 | 中键 / 选中后左键 | 切向体力场（逆时针），同写力路径 |
| Smoke 画笔 | 选中后左键 | 向 `smoke[]` 注入调色板颜色（纯 host，不写力） |
| Eraser 擦除 | 选中后左键 | 向背景灰 `kSmokeBackground` 松弛，消除该处烟雾 |

统一入口 `ApplyToolAt`（圆判据 + 只作用 `ML_FLUID`），`brushStrength` 控制施加/削减量：**1 = 立即将整个笔刷区域设为目标值，0 = 不施加**。`DrawToolRangeMarker` 在鼠标位置绘制浅青色椭圆，标识实际作用范围。

### 7.3 烟雾场（被动标量）

**模型**：每格一个 RGB 颜色（3 个独立被动标量），纯 host 数据（不进 `mrFlow2D`、不参与 GPU 求解，kernel 零改动）。

**实现**（`LbmApp` 成员 `smoke/smokeScratch`，`Step()` 中调 `AdvectSmoke`）：

- **半拉格朗日平流**：`C_new(x,y) = C_old(x - u·dt, y - v·dt)`，双线性插值；按子步拆分（回溯 ≤0.75 格）防糊；无条件稳定；
- **温度扩散**：`smokeTemperature > 0` 时 4 邻域中心差分扩散（显式格式，系数 K ≤ 0.05 保证稳定），无气流也会自然扩散；
- **零速度扰动**：inlet 有流速但某格点速度恰为 (0,0)（驻点/死区）时，加确定性伪随机微小扰动（1e-3），避免烟雾滞留；
- **边界规则**：INLET/OUTLET/WALL 强制置背景灰（入口洁净、出口不堆积），SOLID 保持旧值（烟不穿壁，采样遇固体回退）；
- **初始烟团**：初始为场地中心一团白色软边烟云（非全场铺满），背景浅灰 `(0.82,0.82,0.82)`；
- **擦除**：见 7.2；烟雾视图色条自动隐藏。

文件：`src/testMrLBM2D.cpp`（`AdvectSmoke`、`ApplyToolAt` 注入分支、`Init` 烟场复位、`BuildFieldImage` Smoke 分支）。设计细节见《烟雾场具体实现.md》。

### 7.4 物体系统

- 形状：Circle / Box / Diamond（全局统一，最多 `kMaxBodies=4` 个）；
- 添加：`AddBodyAt`（越界 + 重叠校验，失败弹提示）；删除：`RemoveBody`（换位删除）；
- 拖动：鼠标命中拾取 `PickBody`（容差 radius+3），拖动时壁面速度 = 位移/stepsPerFrame，`ApplyObstaclePose` 处理流固耦合（释放格点取邻域平均矩、固体格点写壁面矩）；
- **放大/缩小校验**（`CanResizeBody`）：新半径不得越界（含 2 格安全间距）、不得与其他物体重合（+1 安全间距）；滑块拖动时沿滑轨取最大合法值，防回弹；
- 键盘/UI 移动写目标点 `tx/ty`，物体平滑逼近。

文件：`src/testMrLBM2D.cpp`（物体系统、UI）+ `src/LbmCases2D.cpp`（形状格点判定）。

### 7.5 预设系统

`ApplyPreset` 内置三档预设（面板 Preset 下拉，选择即应用）：

- **Karman**：默认涡街（有圆柱、有来流）；
- **Jet**：默认射流；
- **Blank**：以 Karman 几何为底的**空白版**——初始/入口速度与扰动全零、无物体，便于从零开始作画/实验。

切换预设 → 填充 `gLoadedParams` → 复用 `ApplyLoadedStartupParams` 分发（三分类：需重建 / 热更新 / 直接生效）。

### 7.6 参数面板

- **直接生效**（写 `definition`，下帧边界/显示即用）：入口 ux（可负）、入口 uy、扰动幅值/周期、各视图颜色上限；
- **热更新**（写 host 再同步设备）：粘度（`SyncFlowScalarsToDevice`）；
- **需重建**（`RestartWithCurrentSettings`）：网格尺寸、初始速度、喷口宽度。

### 7.7 FPS HUD 与 Ctrl 调试

- `Show FPS overlay` 开关（Debug 栏，默认关）：0.5s 滚动窗口统计 FPS，QPC 测量每帧仿真耗时并 EMA 平滑，显示在画面右上角（`DrawFpsOverlay`）；
- 按住 Ctrl：鼠标处出现放大镜（`kMagnifierRadius` 区域放大显示）；Ctrl+左键选中格点，`DrawDebugOverlay` 显示该格点密度/速度/应力等全部数据。

### 7.8 快捷键重构

R 键按下仅记录 `GetTickCount64()`，松开按按住时长分发：**短按（<500ms）Restart**（保留已调参数重建），**长按（≥500ms）Reset**（恢复默认）；失焦时丢弃计时，防误触发。

### 7.9 存档 / 读档

**文件**：`src/ParamsIO.h/.cpp`（`SaveAppParams` / `LoadAppParams`）+ `src/testMrLBM2D.cpp`（对话框与调度）。

- 格式：`key=value` 每行一项，覆盖 case / fieldView / stepsPerFrame / obstacleShape / bodyCount / bodyN{X,Y,Radius} / viscosity / initialUx/Uy / inletUx/Uy / 扰动 / 颜色上限 / jetWidth；
- 读取以当前 case 默认 def 为基底逐项覆盖，数值与物体二次校验（clamp/去重/截断），非法值回退默认，不崩溃；
- **文件对话框**（`PromptSaveParamsFile`/`PromptOpenParamsFile`，`OPENFILENAMEW`）：初始目录 = 可执行目录，中文路径经 CP_ACP 转换；
- **闪退修复**：Windows 文件对话框的模态消息循环会在 ImGui 帧内派发 `WM_PAINT` 触发嵌套 ImGui 帧，导致崩溃。因此按钮只置 `gPendingParamsAction`，对话框统一延迟到 `Render()` 帧末（`ImGui::Render()` 之后、`SwapBuffers` 之前）执行，结果由下一帧弹出成功/失败提示（`gParamsActionResult`）。

---

## 八、功能与文件映射总表

| 功能 | 主要文件 | 说明 |
| --- | --- | --- |
| LBM 内核（D2Q9 重构/碰撞/迭代） | `inc/2D/gpu/mrUtilFuncGpu2D.h`、`inc/2D/gpu/mrLbmSolverGpu2D.cu` | P1-A/B/C，GPU 求解核心（含 §5.5 CUDA 基础优化） |
| 流场数据与求解器管理 | `inc/2D/cpu/mrFlow2D.h`、`inc/2D/cpu/mrSolver2D.h` | 矩存储、双缓冲、kernel 启动 |
| 常量（方向/权重/cs²） | `inc/2D/cpu|gpu/mrConstantParams*2D.h` | D2Q9 模型常量 |
| 算例与边界 | `src/LbmCases2D.h/.cpp` | Karman/Jet 几何、flag、初始化、边界刷新 |
| 参数文件读写 | `src/ParamsIO.h/.cpp` | ini 保存/加载 + 数值/物体二次校验 |
| 程序入口 / 窗口 / 渲染 | `src/testMrLBM2D.cpp` | Win32 + OpenGL + ImGui 主循环 |
| 交互工具（吹风/旋涡） | `src/testMrLBM2D.cpp` | 力注入、力同步与清力 |
| 烟雾场（平流/扩散/画笔/擦除） | `src/testMrLBM2D.cpp` | 被动标量，纯 host |
| 显示系统（视图） | `src/testMrLBM2D.cpp` + `LbmCases2D.cpp` | 四种视图渲染 + 色条；`V` 键在速度/涡量/烟雾间循环，Colorful 仅面板手动进入 |
| 物体系统与流固耦合 | `src/testMrLBM2D.cpp` + `LbmCases2D.cpp` | 拖动/增删/校验/形状 |
| 预设系统 | `src/testMrLBM2D.cpp` | Karman/Jet/Blank |
| FPS HUD / Ctrl 调试 | `src/testMrLBM2D.cpp` | 性能统计、放大镜、格点数据 |
| 存档/读档对话框 | `src/testMrLBM2D.cpp` + `src/ParamsIO.cpp` | 帧末延迟执行（防闪退） |

---

## 九、运行流程小结

启动 → 读 ini → 建窗口 → `LbmApp::Init` 建网格并初始化（含烟雾复位）→ 上传 GPU → 进入消息循环：每 16ms 一帧，未暂停则推进 `stepsPerFrame` 次 LBM 迭代 + 烟雾平流，再生成像素图渲染；鼠标/键盘/面板输入实时作用于流场（力、物体、工具、参数），R 键/预设/读档可随时重建算例，存档可持久化全部参数。
