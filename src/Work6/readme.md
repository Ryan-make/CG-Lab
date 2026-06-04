# 质点-弹簧布料模拟实验

## 📖 实验目标
- 掌握使用 Taichi 框架构建动态 3D 场景与 GGUI 交互面板
- 理解质点-弹簧模型及弹力、阻尼力的计算，处理数值爆炸问题
- 独立编写并对比显式欧拉、半隐式欧拉、隐式欧拉三种积分器的稳定性差异
- 学习 Taichi 中的 `ti.kernel`、`ti.func` 以及 GPU 并行计算的状态同步优化

## 🧮 实验原理
### 质点-弹簧模型
将布料离散化为网格状质点，质点间通过结构弹簧连接。弹力遵循胡克定律：
$$f_{a} = -k_{s} (|x_a - x_b| - l) \frac{x_a - x_b}{|x_a - x_b|}$$
阻尼力：
$$f_{d} = -k_{d} v_{a}$$

### 数值积分方法
根据加速度 $a = F/m$ 和时间步 $\Delta t$ 更新运动状态：
- **显式欧拉**：完全用当前状态预测下一步  
  $$x_{t+1} = x_t + v_t \Delta t,\quad v_{t+1} = v_t + a_t \Delta t$$
- **半隐式欧拉**：先更新速度，再用新速度更新位置  
  $$v_{t+1} = v_t + a_t \Delta t,\quad x_{t+1} = x_t + v_{t+1} \Delta t$$
- **隐式欧拉**：使用下一时刻加速度，通过定点迭代近似求解  
  $$v_{t+1} = v_t + a_{t+1} \Delta t,\quad x_{t+1} = x_t + v_{t+1} \Delta t$$

## ✨ 功能特性
- 可配置的布料网格（默认 20×20），支持重力、弹力、阻尼力模拟
- 三种积分器（显式/半隐式/隐式欧拉）实时切换，直观对比稳定性
- 速度钳制防止数值爆炸
- GGUI 3D 场景实时渲染，支持暂停、重置布料
- 通过 UI 按钮灵活控制模拟状态

## ⚙️ 环境要求
- Python 3.8+
- Taichi ≥ 1.6.0

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install taichi
```

### 2. 运行程序
```bash
python cloth_sim.py
```

### 3. 交互方式
| 按钮 | 功能 |
|------|------|
| Explicit | 切换到显式欧拉积分 |
| Semi-Implicit | 切换到半隐式欧拉积分 |
| Implicit | 切换到隐式欧拉积分（定点迭代） |
| Pause / Resume | 暂停/继续模拟 |
| Reset | 重置布料到初始状态 |

## 🔧 核心实现
- **初始化**：通过多个 `@ti.kernel` 顺序初始化质点位置、速度、弹簧拓扑和渲染索引，保证 GPU 状态同步。
- **力学计算**：使用 `@ti.func` 实现 `compute_forces_on()` 和 `clamp_velocity()`，被积分 Kernel 内联调用以减少 GPU 函数开销。弹簧力累加使用 `ti.atomic_add` 避免写入冲突。
- **积分器**：三种方法分别实现为独立的 `@ti.kernel`（`step_explicit`、`step_semi_implicit`、`step_implicit_iter`），在同一 Kernel 内完成受力计算和状态更新，最小化 Kernel 启动次数。
- **渲染**：基于 `ti.ui.Window` 构建 3D 场景，通过 `window.GUI` 添加按钮实现实时控制。

## 📂 项目结构
```
.
├── cloth_sim.py     # 主程序
└── README.md        # 项目说明
```

## 📝 效果截图
