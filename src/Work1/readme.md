# 旋转与变换

## 项目简介
本项目基于 **Taichi** 图形编程框架，实现三维空间中几何体的 **模型变换（Model）**、**视图变换（View）** 和 **投影变换（Projection）**（统称 MVP 变换），将三维物体渲染到二维屏幕上。  
通过手动推导并编写三个核心矩阵函数，深入理解图形学中的坐标变换流水线，并掌握 Taichi 的基本语法与矩阵操作。

- 基础任务：绘制绕 Z 轴旋转的线框三角形。
- 选做任务：构建立方体并进行透视旋转，增加旋转插值动画。

## 实验目标
1. 理解三维空间中 **模型-视图-投影（MVP）变换** 的完整流程。
2. 独立推导并用 Taichi 代码实现：
   - `get_model_matrix(angle)`：绕 Z 轴的旋转矩阵。
   - `get_view_matrix(eye_pos)`：将相机平移至原点的视图矩阵。
   - `get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)`：透视投影矩阵。
3. 掌握 Taichi 中矩阵构造、向量运算以及 GUI 绘图接口的基本使用。
4. 通过透视除法将齐次坐标映射到标准设备坐标（NDC），并显示线框图形。

## 理论基础与矩阵推导

### 坐标系与约定
- 右手坐标系：X 轴向右，Y 轴向上，Z 轴由屏幕指向外（相机看向 -Z 方向）。
- 点表示为列向量，变换矩阵以左乘方式作用：$\mathbf{v}' = \mathbf{M} \cdot \mathbf{v}$。
- 所有矩阵均为 $4 \times 4$ 齐次坐标形式。

### 模型变换（Model Matrix）
绕 Z 轴旋转角度 $\alpha$（度数）：

$$ M_{\text{model}} = 
\begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 & 0 \\
\sin\alpha &  \cos\alpha & 0 & 0 \\
0          &  0          & 1 & 0 \\
0          &  0          & 0 & 1
\end{bmatrix}
$$

角度需转为弧度：$\alpha_{\text{rad}} = \alpha \cdot \pi / 180$。

### 视图变换（View Matrix）
将相机从位置 $\mathbf{eye} = (e_x, e_y, e_z)$ 平移至原点：

$$ M_{\text{view}} = 
\begin{bmatrix}
1 & 0 & 0 & -e_x \\
0 & 1 & 0 & -e_y \\
0 & 0 & 1 & -e_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 投影变换（Projection Matrix）
分两步：透视→正交挤压 + 正交投影。

- 近平面 Z 坐标 $n = -zNear$，远平面 $f = -zFar$。
- 视锥体上下边界：$t = \tan(\frac{\text{fov}_Y \cdot \pi}{360}) \cdot |n|$，$b = -t$。
- 左右边界：$r = aspect \cdot t$，$l = -r$。

**透视→正交矩阵**：
$$ M_{p2o} = 
\begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

**正交投影矩阵**：
$$ M_{ortho} = 
\begin{bmatrix}
\frac{1}{r} & 0 & 0 & 0 \\
0 & \frac{1}{t} & 0 & 0 \\
0 & 0 & \frac{2}{f-n} & -\frac{f+n}{f-n} \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

最终投影矩阵：$M_{proj} = M_{ortho} \cdot M_{p2o}$。

### MVP 变换与透视除法
$$ \mathbf{v}_{clip} = M_{proj} \cdot M_{view} \cdot M_{model} \cdot \mathbf{v}_{world} $$
透视除法：$x_{ndc} = x_{clip}/w_{clip}$，$y_{ndc} = y_{clip}/w_{clip}$，$z_{ndc} = z_{clip}/w_{clip}$。  
最后视口变换映射到屏幕坐标。

## 代码实现

### 开发环境
- Python 3.8+
- Taichi 1.x
- 运行方式：`python main.py`

### 核心矩阵函数（Taichi 代码）

```python
import taichi as ti
import taichi.math as tm

@ti.func
def get_model_matrix(angle: ti.f32) -> ti.Matrix:
    rad = angle * tm.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0,0.0,1.0, 0.0],
        [0.0,0.0,0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos: ti.Vector) -> ti.Matrix:
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32,
                          zNear: ti.f32, zFar: ti.f32) -> ti.Matrix:
    rad = eye_fov * tm.pi / 180.0
    t = ti.tan(rad / 2.0) * ti.abs(zNear)
    r = aspect_ratio * t
    n = -zNear
    f = -zFar

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho = ti.Matrix([
        [1.0/r, 0.0, 0.0, 0.0],
        [0.0, 1.0/t, 0.0, 0.0],
        [0.0, 0.0, 2.0/(f - n), -(f + n)/(f - n)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return M_ortho @ M_p2o
```

### 顶点处理与绘制
```python
for v in vertices:
    v_homo = ti.Vector([v[0], v[1], v[2], 1.0])
    v_clip = proj @ view @ model @ v_homo
    v_ndc = ti.Vector([v_clip[0]/v_clip[3], v_clip[1]/v_clip[3]])
    screen_x = int((v_ndc[0] + 1.0) * 0.5 * width)
    screen_y = int((v_ndc[1] + 1.0) * 0.5 * height)
```
使用 `gui.line()` 依次连接三角形三边，形成线框三角形。

## 运行与效果
1. 运行 `main.py`，窗口显示白色线框三角形，初始顶点 $(2,0,-2)$、$(0,2,-2)$、$(-2,0,-2)$。
2. 三角形绕 Z 轴旋转，角度随时间变化。
3. 视图变换将相机置于原点，三角形呈现近大远小的透视效果。

## 视频展示


## 选做内容

### 立方体线框渲染
- **顶点**：以原点为中心，边长 2 的正方体，8 个顶点坐标均为 $\pm1$ 组合。
- **边**：预定义 12 条边的顶点索引对。
- **旋转**：使用绕 X、Y、Z 任意轴的旋转矩阵，组合实现空间旋转感。

```python
def get_rotation_y(angle):
    rad = angle * tm.pi / 180.0
    c, s = ti.cos(rad), ti.sin(rad)
    return ti.Matrix([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s,0, c, 0],
        [0, 0, 0, 1]
    ])
```

### 旋转插值
- 定义起始姿态（单位矩阵）和目标姿态（如绕 Y 轴转 90°）。
- 使用线性插值 + Gram-Schmidt 正交化，或四元数 Slerp，在两个姿态间平滑过渡。
- 程序同时显示两个立方体，颜色不同，随时间呈现过渡动画。

## 常见问题与解决
1. **角度未转弧度**：`ti.sin`、`ti.tan` 需弧度输入，必须乘以 $\pi/180$。
2. **投影 Z 方向错误**：正确设置 $n = -zNear$，$f = -zFar$，并推导正交矩阵的 Z 映射。
3. **视口变换越界**：NDC 到屏幕映射应为 `(ndc + 1) * 0.5 * screen_dim`。
4. **立方体边连接错误**：明确 12 条边的顶点索引对，避免绘制缺失。

##视频展示




## 总结
本实验完整实践了三维渲染管线的核心变换流程。从旋转、平移矩阵到透视投影的推导，每一步都加深了对空间坐标变换的理解。Taichi 的向量化矩阵操作让公式与代码高度一致，降低了实现难度。在此基础上扩展的立方体渲染与姿态插值，进一步展现了三维图形程序的乐趣。

## 参考资料
- GAMES101 - 现代计算机图形学入门，Lec 4-5 变换与投影
- Taichi 官方文档：https://docs.taichi-lang.org/
- 《Fundamentals of Computer Graphics》Chapter 7 Viewing

