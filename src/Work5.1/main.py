import taichi as ti
import math

# 初始化 Taichi，指定 CPU 架构以确保跨平台兼容性
ti.init(arch=ti.cpu)

# 渲染分辨率
res = 256

# 缓冲区定义
target_pixels = ti.field(dtype=ti.f32, shape=(res, res))
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res))

# 标量 Loss 与待优化的三维光源位置（均需开启梯度追踪）
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)

# 场景几何参数与目标设定
sphere_center = ti.Vector([0.5, 0.5, 0.5])
sphere_radius = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]


@ti.kernel
def generate_target():
    """生成目标参考图像 (Ground Truth)"""
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            
            target_light_vec = ti.Vector(TARGET_LIGHT)
            l_dir = (target_light_vec - p).normalized()

            # 标准 Lambertian 漫反射
            dot_val = n.dot(l_dir)
            target_pixels[i, j] = ti.max(0.0, ti.min(1.0, dot_val))
        else:
            target_pixels[i, j] = 0.0


@ti.kernel
def render_and_compute_loss():
    """执行正向渲染并计算允许梯度回传的均方误差 (MSE Loss)"""
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        intensity = 0.0
        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            l_dir = (light_pos[None] - p).normalized()

            dot_val = n.dot(l_dir)
            
            # 使用无分支的 ti.max 实现 Leaky Lambertian 模型
            # 引入 0.1 的泄漏系数，确保处于阴影中的光源也能产生微小的非零梯度
            # 注：此处必须保留负值用于计算 Loss，不可提前 Clamp 到 0
            intensity = ti.max(0.1 * dot_val, dot_val)
        
        # 累加均方误差
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (res * res)) * (diff ** 2)
        
        # 将左半侧设为目标图像，右半侧设为当前渲染结果
        # 在显示输出层面进行物理限幅，保证 GUI 正常渲染
        display_pixels[i, j] = target_pixels[i, j]
        display_pixels[i + res, j] = ti.max(0.0, ti.min(1.0, intensity))


def main():
    # 1. 场景初始化
    generate_target()
    
    # 设定偏离目标的初始光源位置（位于球体偏背面区域）
    light_pos[None] = [0.2, 0.2, 0.8]  
    
    # 2. Adam 优化器超参数设定
    m = [0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0]
    beta1 = 0.9
    beta2 = 0.999
    lr = 0.02
    eps = 1e-8

    # 3. GUI 初始化
    gui = ti.GUI("Differentiable Rendering (Left: Target, Right: Current)", res=(res * 2, res))

    print(f"Target Light Position: {TARGET_LIGHT}")
    print(f"Initial Light Position: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print("-" * 40)

    # 4. 优化主循环
    for iter in range(1, 301):
        # 每轮迭代前清空损失值
        loss[None] = 0.0
        
        # 记录计算图，正向执行后自动反向传播计算参数梯度
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()

        grad = light_pos.grad[None]

        # 运用 Adam 算法更新光源三维坐标
        for c in range(3):
            m[c] = beta1 * m[c] + (1 - beta1) * grad[c]
            v[c] = beta2 * v[c] + (1 - beta2) * grad[c] * grad[c]
            
            m_hat = m[c] / (1 - beta1**iter)
            v_hat = v[c] / (1 - beta2**iter)
            
            light_pos[None][c] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        # 日志输出
        if iter % 10 == 0:
            pos = light_pos[None]
            print(f"Iter {iter:03d} | Loss: {loss[None]:.6f} | "
                  f"Light Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # 刷新可视化窗口
        gui.set_image(display_pixels)
        gui.show()

if __name__ == "__main__":
    main()
