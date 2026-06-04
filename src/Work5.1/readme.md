# 可微光栅化与网格变形实验（低难度）

## 📖 实验目标
- 理解可微光栅化在处理离散网格边界时的数学近似方法
- 掌握通过多视角二维剪影反推并优化三维网格顶点坐标
- 深刻理解正则化在防止拓扑崩坏和陷入局部最优中的关键作用

## 🧮 实验原理
将初始球体通过梯度下降逐步形变为目标奶牛形状，需要解决两个核心问题：

### 1. 软光栅化 (Soft Rasterization) — 防梯度消失
传统硬光栅化边界处梯度为零。软光栅化利用像素到三角形边缘的距离 $d$ 和 Sigmoid 函数，产生平滑的概率过渡：
$$A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right)$$
其中 $\sigma$ 控制模糊程度，即使顶点在像素外部也能提供非零梯度。

### 2. 网格正则化 (Mesh Regularization) — 防局部最优
仅靠图像损失会使顶点交叉重叠。引入三种正则化损失保持网格光滑：
- **拉普拉斯平滑**：约束相邻顶点，避免尖锐突起。
- **边长一致性**：惩罚过长或过短的边，防止三角形严重拉伸。
- **法线一致性**：约束相邻面法线方向接近。

总损失函数：
$$L_{total} = L_{silhouette} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal}$$

## ✨ 功能特性
- 加载目标奶牛网格，从多视角渲染参考剪影图
- 初始化高细分球体作为源网格
- 基于 PyTorch3D 的软剪影光栅化器进行可微渲染
- 联合优化剪影损失与三种几何正则化项
- 可视化球体逐步形变为奶牛的过程

## ⚙️ 环境要求
- Python 3.8+
- PyTorch ≥ 1.12
- TorchVision
- PyTorch3D ≥ 0.7

## 🚀 快速开始

### 1. 安装依赖
建议使用 Conda 创建环境（Windows / macOS 均可）：
```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
# 根据 CUDA 版本或 CPU 安装 PyTorch，例如：
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# 安装 PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pytorch3d
```

### 2. 运行程序
```bash
python main.py
```

### 3. 输出
程序将在优化循环中实时或定期显示当前形变网格的剪影与目标剪影的对比，并最终保存/展示形变结果。

## 🔧 核心实现
- **目标视图生成**：加载目标网格，在固定多视角下渲染剪影图。
- **可微光栅化**：使用 PyTorch3D 的 `SoftSilhouetteShader`，设置合适的 `sigma` 参数。
- **优化变量**：将源球体的顶点偏移量 `deform_verts` 设为可训练参数。
- **损失函数**：
  - 剪影损失：当前视图剪影与目标剪影的 MSE。
  - 拉普拉斯损失：使用 PyTorch3D 的 `mesh_laplacian_smoothing`。
  - 边长损失：`mesh_edge_loss` 惩罚边长方差。
  - 法线损失：`mesh_normal_consistency`。
- **优化器**：使用 Adam 更新顶点偏移，学习率可调。

## 📂 项目结构
```
.
├── main.py  # 主程序
└── README.md             # 项目说明
```

## 📝 效果展示
<img width="480" height="300" alt="实验六（低难度）" src="https://github.com/user-attachments/assets/b897d34d-b3a3-48ff-82b5-283df25aee2c" />
