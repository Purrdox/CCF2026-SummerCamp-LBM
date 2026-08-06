# Home2D：二维 LBM CUDA 编程作业

本项目是一个二维格子 Boltzmann 方法（Lattice Boltzmann Method，LBM）教学作业。程序使用 D2Q9 离散速度模型和 CUDA 完成流场推进，并使用 OpenGL 实时显示计算结果。

## 1. 作业内容

作业内容分为四部分：

1. 实现最核心的二维 LBM solver。
2. 实现 Kármán Vortex 代码。
3. 实现 Jet Flow 代码。
4. 实现两个 Case 共享的边界分类、初始化和边界刷新代码。

OpenGL 窗口、颜色映射、键盘交互、CUDA 内存管理和大部分公共辅助代码均已提供，同学们只需要补全带有以下标记的代码块：

```cpp
// ===== ASSIGNMENT FILL BEGIN: ... =====
// TODO: ...
// ===== ASSIGNMENT FILL END: ... =====
```







## 2. 工程环境

### 2.1 必需环境

- Windows 10 或 Windows 11。
- Visual Studio 2022。
- Visual Studio 工作负载“使用 C++ 的桌面开发”。
- MSVC 平台工具集 `v143`。
- Windows 10/11 SDK。
- NVIDIA CUDA Toolkit 12.1。
- 受当前 CUDA Toolkit 和驱动支持的 NVIDIA GPU。
- 支持 OpenGL 的显卡驱动。

工程默认配置为 `Release|x64`。建议统一使用 x64，不要使用 Win32 配置。

### 2.2 计算后端

- LBM 迭代由 CUDA GPU 完成。
- CPU 负责算例初始化、边界更新、移动固体处理和显示数据准备。
- OpenGL 负责绘制结果。

注意，当前项目没有可替代 CUDA kernel 的 CPU 迭代后端。因此，没有 NVIDIA GPU 或 CUDA 环境时，可以阅读和编译部分普通 C++ 代码，但不能正常运行完整仿真。

## 3. 目录与代码框架

```text
Home2d/
|-- LBM_MRLBM.sln
|-- LBM_MRLBM.vcxproj
|-- README.md
|-- 3rdParty/
|   `-- CUDA 辅助头文件
|-- common/
|   |-- 公共数据类型、颜色映射和 CUDA 包装
|   `-- ...
|-- inc/
|   `-- 2D/
|       |-- cpu/
|       |   |-- mrFlow2D.h
|       |   |-- mrSolver2D.h
|       |   |-- mrConstantParamsCpu2D.h
|       |   `-- mlInit2D.h
|       `-- gpu/
|           |-- mrConstantParamsGpu2D.h
|           |-- mrUtilFuncGpu2D.h
|           |-- mrLbmSolverGpu2D.h
|           `-- mrLbmSolverGpu2D.cu
`-- src/
    |-- LbmCases2D.h
    |-- LbmCases2D.cpp
    `-- testMrLBM2D.cpp
```

### 3.1 主要文件

| 文件 | 作用 | 是否包含代码填空 |
| --- | --- | --- |
| `inc/2D/cpu/mrFlow2D.h` | 定义二维流场、矩数组、边界标记、外力和网格参数 | 否 |
| `inc/2D/cpu/mrSolver2D.h` | 管理 host/device 数据，调用 GPU solver | 否 |
| `inc/2D/gpu/mrConstantParamsGpu2D.h` | 定义 D2Q9 方向、权重和声速平方 | 否 |
| `inc/2D/gpu/mrUtilFuncGpu2D.h` | 分布函数重构和碰撞辅助函数 | 是，`P1-A`、`P1-B` |
| `inc/2D/gpu/mrLbmSolverGpu2D.cu` | CUDA LBM kernel、双缓冲交换和 kernel launch | 是，`P1-C` |
| `src/LbmCases2D.h` | 定义算例编号、参数结构和公共接口 | 否 |
| `src/LbmCases2D.cpp` | 算例几何、边界标记、初始化和边界更新 | 是，`P2-A`、`P3-A`、`P2-P3-A`、`P2-P3-B`、`P2-P3-C` |
| `src/testMrLBM2D.cpp` | 程序入口、移动固体、数据同步、OpenGL 和键盘交互 | 是，`P2-B` |

### 3.2 数据流

一次完整的程序运行过程如下：

```text
DemoCaseDefinition
        |
        v
创建 mrFlow2D 网格
        |
        v
InitializeDemoCase：设置 flag 和初始矩
        |
        v
mrSolver2D::mlTransData2Gpu：上传 GPU
        |
        v
RefreshDemoCaseBoundaries：更新 host 边界
        |
        v
同步边界到 GPU
        |
        v
mrSolver2D::mlIterateGpu
        |
        v
mrSolver2DKernel：pull streaming + collision
        |
        v
交换 fMom / fMomPost
        |
        v
复制矩到 host，生成颜色图并由 OpenGL 绘制
```

`steps/frame` 表示每个 UI frame 内执行多少个 LBM 迭代，请注意窗口帧数与 LBM 迭代次数不是同一个概念。

## 4. 数值模型与数据表示

### 4.1 D2Q9 离散速度

项目使用以下方向编号：

| `i` | `ex[i]` | `ey[i]` | 含义 |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | 静止 |
| 1 | 1 | 0 | 右 |
| 2 | 0 | 1 | 上 |
| 3 | -1 | 0 | 左 |
| 4 | 0 | -1 | 下 |
| 5 | 1 | 1 | 右上 |
| 6 | -1 | 1 | 左上 |
| 7 | -1 | -1 | 左下 |
| 8 | 1 | -1 | 右下 |

对应权重为：

```text
w0 = 4/9
w1 = w2 = w3 = w4 = 1/9
w5 = w6 = w7 = w8 = 1/36
cs^2 = 1/3
```

CPU 和 GPU 常量分别位于：

- `inc/2D/cpu/mrConstantParamsCpu2D.h`
- `inc/2D/gpu/mrConstantParamsGpu2D.h`

### 4.2 六个存储矩

本项目没有为每个格点长期存储 9 个分布函数，而是在 `fMom` 和 `fMomPost` 中存储 6 个矩：

| 偏移 | 内容 |
| ---: | --- |
| 0 | 密度 `rho` |
| 1 | x 方向速度 `ux` |
| 2 | y 方向速度 `uy` |
| 3 | 归一化二阶矩 `pixx` |
| 4 | 归一化二阶矩 `piyy` |
| 5 | 归一化二阶矩 `pixy` |

每个格点的起始位置为 `index * 6`。`fMom` 是当前时刻，`fMomPost` 是下一时刻。一次 GPU 更新完成后，已提供的 `MomSwap` 会交换两个指针。

### 4.3 格点标记

常用标记包括：

- `ML_FLUID`：由 CUDA kernel 更新的流体格点。
- `ML_INLET`：入口边界。
- `ML_OUTLET`：开放出口。
- `ML_WALL_LEFT`、`ML_WALL_RIGHT`、`ML_WALL_UP`、`ML_WALL_DOWN`：固壁。
- `ML_SOLID`：固体格点。

### 4.4 坐标方向

网格采用：

- `x = 0` 位于画面左侧。
- `y = 0` 位于画面底部。
- 正 `y` 方向指向画面上方。

两个算例的主流方向都是从下向上。

## 5. 需要填写的代码

需要补全所有 `ASSIGNMENT FILL BEGIN/END` 之间的代码，共 9 处。

| 部分 | 标记 | 函数 | 文件路径 |
| --- | --- | --- | --- |
| LBM solver | `P1-A` | `mrUtilFuncGpu2D::mlCalDistributionD2Q9AtIndex` | `inc/2D/gpu/mrUtilFuncGpu2D.h` |
| LBM solver | `P1-B` | `mrUtilFuncGpu2D::mlGetPIAfterCollision` | `inc/2D/gpu/mrUtilFuncGpu2D.h` |
| LBM solver | `P1-C` | `mrSolver2DKernel` | `inc/2D/gpu/mrLbmSolverGpu2D.cu` |
| Kármán Vortex | `P2-A` | `IsDemoCaseObstacleCell` | `src/LbmCases2D.cpp` |
| Kármán Vortex | `P2-B` | `LbmApp::ApplyObstaclePose` 中的标记块 | `src/testMrLBM2D.cpp` |
| Jet Flow | `P3-A` | `IsJetInlet` | `src/LbmCases2D.cpp` |
| 两个 Case 共享 | `P2-P3-A` | `GetDemoCaseBaseFlag` | `src/LbmCases2D.cpp` |
| 两个 Case 共享 | `P2-P3-B` | `InitializeDemoCase` | `src/LbmCases2D.cpp` |
| 两个 Case 共享 | `P2-P3-C` | `RefreshDemoCaseBoundaries` | `src/LbmCases2D.cpp` |

`P2-P3-*` 是 Kármán Vortex 和 Jet Flow 共享的函数，只填写一次，并在函数中分别处理两个 Case。

### P1：实现核心 LBM Solver

本题分为三个部分：

#### P1-A `mrUtilFuncGpu2D::mlCalDistributionD2Q9AtIndex`

根据密度、速度和归一化二阶矩，重构指定方向的 D2Q9 分布函数。

#### P1-B `mrUtilFuncGpu2D::mlGetPIAfterCollision`

对三个原始二阶矩原地执行 BGK 碰撞，并加入体力修正。

#### P1-C `mrSolver2DKernel`

对流体格点实现一次完整的 LBM 更新，包括 pull streaming、带体力修正的宏观量恢复、二阶矩碰撞、归一化和结果写回。

### P2：实现 Kármán Vortex

Kármán vortex 算例描述均匀来流绕过圆柱后形成的尾流。随着 Reynolds 数增大，对称尾流会失稳并产生交替涡脱落。

#### 算例参数

| 参数 | 数值 |
| --- | ---: |
| 网格 | `96 x 192` |
| 初始速度 | `u = (0, 0.10)` |
| 运动黏度 | `nu = 0.008` |
| 圆柱初始中心 | `(47.5, 45.5)` |
| 圆柱半径 | `R = 8` |
| 特征长度 | `D = 16` |
| Reynolds 数 | `Re = 200` |
| 入口横向扰动幅值 | `0.004` |
| 扰动周期 | `320` 个 LBM 迭代 |
| 默认 LBM 步数/UI frame | `15` |
| 默认显示 | 有符号涡量 |

Reynolds 数为：

```text
Re = U * D / nu = 0.10 * 16 / 0.008 = 200
```



本题包含两个 Kármán Vortex 专属部分：

#### P2-A `IsDemoCaseObstacleCell`

判断格点是否被当前圆柱覆盖；不含障碍物的 Case 应返回 `false`。

#### P2-B `LbmApp::ApplyObstaclePose`

完成圆柱移动时的流固耦合：重新划分固体区域，重建新释放流体格点的矩，并为当前固体格点设置壁面速度和非平衡应力。

### P3：实现 Jet Flow

Jet flow 算例描述底部窄缝向静止流体区域喷射。高速核心两侧形成剪切层，并逐渐卷吸周围低速流体。

#### 算例参数

| 参数 | 数值 |
| --- | ---: |
| 网格 | `96 x 192` |
| 喷口速度 | `u = (0, 0.08)` |
| 喷口宽度 | `D = 16` 个格点 |
| 运动黏度 | `nu = 0.020` |
| Reynolds 数 | `Re = 64` |
| 顶部边界 | 开放出口 |
| 左右边界 | 无滑移固壁 |
| 默认 LBM 步数/UI frame | `15` |
| 默认显示 | 速度模长 |



本题包含一个 Jet Flow 专属部分：

#### P3-A `IsJetInlet`

判断底边格点是否位于居中的喷口范围，并保证喷口宽度恰好为 `jetWidth`。

### P2-P3：实现两个 Case 的共享部分

本题分为三个部分，每个函数都需要同时处理 Kármán Vortex 和 Jet Flow。

#### P2-P3-A `GetDemoCaseBaseFlag`

设置两个 Case 的固定边界和内部流体区域：Kármán 使用底部入口及顶部、左右开放出口；Jet Flow 使用底部中央喷口、底部其余固壁、左右固壁和顶部出口。

#### P2-P3-B `InitializeDemoCase`

设置全部格点的边界或固体标记，选择对应的初始、入口或固壁速度，并以 `rho = 1` 的平衡态初始化两个矩缓冲区。

#### P2-P3-C `RefreshDemoCaseBoundaries`

逐帧更新外层边界矩：入口使用规定速度，开放出口复制相邻内部速度，固壁使用零速度，并从相邻内部格点外推非平衡应力。Kármán Vortex 还需要在入口加入横向扰动。

## 6. 编译步骤

### 6.1 使用 Developer PowerShell

推荐使用 Visual Studio 自带的 `Developer PowerShell for VS 2022`。

1. 打开 Developer PowerShell。
2. 进入仓库中的工程目录：

   ```powershell
   cd Home2d
   ```

3. 编译 `Release|x64`：

   ```powershell
   msbuild LBM_MRLBM.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
   ```

4. 如果需要完整重新编译：

   ```powershell
   msbuild LBM_MRLBM.sln /m /t:Rebuild /p:Configuration=Release /p:Platform=x64
   ```

5. 编译成功后，可执行文件通常位于：

   ```text
   x64/Release/LBM_MRLBM.exe
   ```

6. 查看命令行帮助：

   ```powershell
   .\x64\Release\LBM_MRLBM.exe --help
   ```

### 6.2 使用 Visual Studio

1. 用 Visual Studio 2022 打开 `LBM_MRLBM.sln`。
2. 在工具栏中选择 `Release`。
3. 平台选择 `x64`。
4. 选择“生成 > 生成解决方案”。
5. 确认输出窗口最后显示生成成功。
6. 使用“调试 > 开始执行（不调试）”运行，或从终端运行生成的可执行文件。



## 7. 平台工具集与 CUDA 版本

### 7.1 找不到 `v143`

典型错误：

```text
error MSB8020: 无法找到平台工具集 v143
```

处理方法：

1. 通过 Visual Studio Installer 安装“使用 C++ 的桌面开发”和对应 MSVC 工具集。
2. 或在 Visual Studio 中右键项目，选择“属性 > 配置属性 > 常规”，将平台工具集改为本机版本。
3. 也可以修改 `LBM_MRLBM.vcxproj` 中的：

   ```xml
   <PlatformToolset>v143</PlatformToolset>
   ```

VS2019 通常使用 `v142`，VS2022 通常使用 `v143`。如果修改工程文件，应检查全部需要使用的 x64 配置。

### 7.2 CUDA Build Customizations 不匹配

先检查本机 CUDA：

```powershell
nvcc --version
```

工程当前引用：

```xml
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.1.props" />
...
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.1.targets" />
```

如果本机安装其他 CUDA 版本，需要让这两处版本号保持一致，或在 Visual Studio 的“生成依赖项 > 生成自定义”中选择本机 CUDA 版本。


## 8. 运行算例



默认启动：

```powershell
.\x64\Release\LBM_MRLBM.exe
```

或者显式选择算例，如：

```powershell
.\x64\Release\LBM_MRLBM.exe --case karman
```

或

```powershell
.\x64\Release\LBM_MRLBM.exe --case jetflow
```

程序运行后，也可以按 `1` 或 `2` 重建并切换算例。

## 9. 键盘交互

| 按键 | 功能 |
| --- | --- |
| `1` | 切换到 Kármán vortex |
| `2` | 切换到 Jet flow |
| `V` | 切换可视化速度模长 / 带符号涡量 |
| `↑` / `↓` / `←` / `→` | 移动 Kármán vortex 圆柱障碍物位置 |
| `Space` | 暂停或继续 |
| `S` | 推进一个 UI frame，在暂停后使用，便于观察每一帧的步进 |
| `+` / `-` | 调整每帧 LBM 步数 |
| `R` | 重置当前算例 |
| `Esc` | 退出 |
