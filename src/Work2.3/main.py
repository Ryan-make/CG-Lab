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

# 曲线坐标 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法（贝塞尔）"""
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

def cubic_uniform_b_spline(control_points):
    """均匀三次B样条曲线计算（核心实现）"""
    n = len(control_points)
    curve_points = []
    
    # 三次B样条基矩阵（标准均匀三次B样条）
    M = np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ], dtype=np.float32) / 6.0
    
    # 不足4个点，无法绘制B样条
    if n < 4:
        return np.array(curve_points, dtype=np.float32)
    
    # 每4个相邻控制点生成一段曲线，总共有 n-3 段
    num_segments = n - 3
    samples_per_segment = NUM_SEGMENTS // num_segments
    
    # 遍历每一段
    for seg in range(num_segments):
        # 取当前段的4个控制点
        p0 = control_points[seg]
        p1 = control_points[seg+1]
        p2 = control_points[seg+2]
        p3 = control_points[seg+3]
        P = np.array([p0, p1, p2, p3], dtype=np.float32)
        
        # 对当前段进行采样
        for i in range(samples_per_segment):
            t = i / samples_per_segment
            t_vec = np.array([t**3, t**2, t, 1], dtype=np.float32)
            # 矩阵乘法计算曲线上的点
            pt = t_vec @ M @ P
            curve_points.append(pt)
    
    # 转换为numpy数组
    return np.array(curve_points, dtype=np.float32)

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """GPU并行绘制曲线"""
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier <-> B-Spline (Press B to switch)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    use_bspline = False  # False=贝塞尔模式，True=B样条模式
    
    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            # 鼠标左键添加控制点
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            # C键清空画布
            elif e.key == 'c': 
                control_points = []
                print("Canvas cleared.")
            # B键切换曲线模式
            elif e.key == 'b':
                use_bspline = not use_bspline
                mode = "B-Spline" if use_bspline else "Bezier"
                print(f"Switched to: {mode} mode")
        
        clear_pixels()
        current_count = len(control_points)
        
        # 绘制曲线
        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            
            if not use_bspline:
                # 贝塞尔曲线绘制
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_points_np[t_int] = de_casteljau(control_points, t)
            else:
                # B样条曲线绘制
                b_spline_points = cubic_uniform_b_spline(control_points)
                if len(b_spline_points) > 0:
                    # 截取/填充到固定采样数
                    take = min(len(b_spline_points), NUM_SEGMENTS + 1)
                    curve_points_np[:take] = b_spline_points[:take]
            
            # 发送到GPU并绘制
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)
        
        # 绘制控制点与控制线
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