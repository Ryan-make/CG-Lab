import taichi as ti
import numpy as np

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 【性能优化核心 1】：新增一个用于存放曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# ====================== 【反走样核心代码】 ======================
@ti.kernel
def draw_curve_antialiased(n: ti.i32):
    # 遍历所有曲线采样点
    for i in range(n):
        pt = curve_points_field[i]
        # 亚像素坐标：0~1 浮点数，保留小数精度
        x_sub = pt[0] * WIDTH
        y_sub = pt[1] * HEIGHT

        # 取中心整数坐标
        cx = ti.cast(x_sub, ti.i32)
        cy = ti.cast(y_sub, ti.i32)

        # 遍历 3x3 邻域（反走样关键）
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                x = cx + dx
                y = cy + dy
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # 计算像素中心到亚像素点的欧氏距离
                    dist_x = (x + 0.5) - x_sub
                    dist_y = (y + 0.5) - y_sub
                    dist = ti.sqrt(dist_x**2 + dist_y**2)

                    # 距离衰减函数：距离越近，亮度越高
                    # 阈值 1.5 保证平滑过渡
                    strength = ti.max(0.0, 1.0 - dist / 1.5)

                    # 叠加绿色（叠加方式避免过曝）
                    pixels[x, y] += ti.Vector([0.0, strength, 0.0]) * 0.4
# =================================================================

def main():
    window = ti.ui.Window("Antialiased Bezier Curve", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            elif e.key == 'c': 
                control_points = []
                print("Canvas cleared.")
        
        clear_pixels()
        
        current_count = len(control_points)
        if current_count >= 2:
            # 1. 在 CPU 端把所有点算好
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            # 2. 一次性传给GPU
            curve_points_field.from_numpy(curve_points_np)
            
            # 3. 调用【反走样】绘制函数
            draw_curve_antialiased(NUM_SEGMENTS + 1)
                    
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()