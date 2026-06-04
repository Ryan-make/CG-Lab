# Phong 光照模型实验

## 📖 实验目标
- 理解局部光照的基本原理，区分环境光、漫反射和镜面高光
- 掌握三维空间中的向量运算（法向量、光线方向、视线方向与反射向量）
- 利用 Taichi 实现交互式渲染，通过 UI 控件实时调节材质参数

## 🧮 实验原理
Phong 光照模型将物体表面反射光分为三个分量，叠加得到最终颜色：

$$I = I_{ambient} + I_{diffuse} + I_{specular}$$

- **环境光 (Ambient)**：模拟均匀背景光  
  $$I_{ambient} = K_a \times C_{light} \times C_{object}$$
- **漫反射 (Diffuse)**：模拟粗糙表面散射光，遵循 Lambert 定律  
  $$I_{diffuse} = K_d \times \max(0, \mathbf{N} \cdot \mathbf{L}) \times C_{light} \times C_{object}$$
- **镜面高光 (Specular)**：模拟光滑表面反射的强光  
  $$I_{specular} = K_s \times \max(0, \mathbf{R} \cdot \mathbf{V})^n \times C_{light}$$

其中 $\mathbf{N}$ 为表面法向量，$\mathbf{L}$ 为指向光源的方向，$\mathbf{V}$ 为指向摄像机的方向，$\mathbf{R}$ 为反射方向，$n$ 为高光指数。

## ✨ 功能特性
- 代码驱动场景：包含红色球体和紫色圆锥两个隐式几何体
- 光线投射 (Ray Casting) 与深度测试，正确实现遮挡
- 完整 Phong 光照着色（Ambient + Diffuse + Specular）
- 四个 UI 滑动条实时调节材质参数：
  - Ka（环境光系数）
  - Kd（漫反射系数）
  - Ks（镜面高光系数）
  - Shininess（高光指数）
- 可选升级 Blinn-Phong 模型与硬阴影（选做）

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
python main.py
```

### 3. 交互方式
使用 GUI 窗口下方的四个滑块实时调整材质参数，观察光照效果变化。

## 🔧 核心实现
- **场景定义**：在 Taichi 内核中用数学隐式函数定义球体和圆锥，摄像机位于 (0,0,5)，点光源位于 (2,3,4)
- **光线求交**：为每个像素发射射线，分别计算与球体和圆锥的交点，取最小正数 $t$ 得到最近交点
- **法向量计算**：在交点处通过解析公式计算单位法向量
- **Phong 着色**：计算 $\mathbf{L}$、$\mathbf{V}$、$\mathbf{R}$ 并归一化，分通道累加环境光、漫反射和高光，最后用 `clamp(color, 0.0, 1.0)` 防止过曝
- **UI 控件**：使用 `ti.ui.Window` 和 `slider` 绑定参数，实现实时更新渲染

## 🎯 选做内容
### 1. Blinn-Phong 模型
- 使用半程向量 $\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{|\mathbf{L} + \mathbf{V}|}$ 替代反射向量 $\mathbf{R}$
- 高光计算改为 $(\mathbf{N} \cdot \mathbf{H})^n$，在大入射角时高光区域更柔和、自然

### 2. 硬阴影
- 从交点向光源方向发射阴影射线
- 若在到达光源前与其他物体相交，则该点处于阴影中，仅保留环境光分量

## 📂 项目结构
```
.
├── phong.py          # 主程序
└── README.md         # 项目说明
```

## 📝 效果展示
必做：

<img width="480" height="387" alt="实验四" src="https://github.com/user-attachments/assets/cacaa661-fdd7-4ad5-8940-a4e5f78b29c9" />


选做：

<img width="480" height="387" alt="实验四 选做" src="https://github.com/user-attachments/assets/890d7e5b-74a7-443b-8201-570d7827d356" />



