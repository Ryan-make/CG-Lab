import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# ===================== 立方体参数 =====================
CUBE_VERTICES = 8
CUBE_EDGES = 12

vertices = ti.Vector.field(3, dtype=ti.f32, shape=CUBE_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=CUBE_VERTICES)
cube_edge_indices = ti.Vector.field(2, dtype=ti.i32, shape=CUBE_EDGES)

# ===================== 变换矩阵 =====================
@ti.func
def get_model_matrix(angle_y: float, angle_x: float):
    """模型矩阵：同时支持 Y轴（左右）+ X轴（上下）旋转"""
    # Y 轴旋转（左右）
    rad_y = angle_y * math.pi / 180.0
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)
    
    # X 轴旋转（上下）
    rad_x = angle_x * math.pi / 180.0
    cx = ti.cos(rad_x)
    sx = ti.sin(rad_x)

    # 绕 Y 轴旋转矩阵
    rot_y = ti.Matrix([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 绕 X 轴旋转矩阵
    rot_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 组合旋转：先X后Y，3D效果最自然
    return (rot_y @ rot_x).cast(ti.f32)

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ]).cast(ti.f32)

@ti.func
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float):
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ]).cast(ti.f32)
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]).cast(ti.f32)
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]).cast(ti.f32)
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

# ===================== 顶点变换 =====================
@ti.kernel
def compute_transform(angle_y: float, angle_x: float):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle_y, angle_x)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(CUBE_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ===================== 初始化立方体 =====================
def init_cube():
    cube_points = [
        (-1, -1, -1), (1, -1, -1),
        (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1),
        (1, 1, 1), (-1, 1, 1)
    ]
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    for i in range(CUBE_VERTICES):
        vertices[i] = cube_points[i]
    
    for i in range(CUBE_EDGES):
        cube_edge_indices[i] = edges[i]

# ===================== 主程序（新增插值 + 双姿态） =====================
def main():
    init_cube()
    gui = ti.GUI("3D立方体 - 姿态插值过渡", res=(700, 700))
    
    # ===================== 【新增】定义两个不同姿态 =====================
    # 姿态 A：初始角度（正面）
    angle_y_a, angle_x_a = 0.0, 0.0
    # 姿态 B：目标角度（斜侧面）
    angle_y_b, angle_x_b = 120.0, 45.0
    
    # 插值参数
    t = 0.0               # 插值因子 [0,1]
    speed = 0.01          # 过渡速度
    forward = True        # 正向/反向切换
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            # 空格：手动切换姿态
            if gui.event.key == ti.GUI.SPACE:
                forward = not forward
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # ===================== 【核心】旋转插值计算 =====================
        if forward:
            t += speed
            if t >= 1.0:
                t = 1.0
        else:
            t -= speed
            if t <= 0.0:
                t = 0.0
        
        # 线性插值：current = A + t*(B-A)
        current_angle_y = angle_y_a + t * (angle_y_b - angle_y_a)
        current_angle_x = angle_x_a + t * (angle_x_b - angle_x_a)
        
        # 计算变换
        compute_transform(current_angle_y, current_angle_x)
        
        # 绘制立方体线框
        for i in range(CUBE_EDGES):
            idx0, idx1 = cube_edge_indices[i]
            gui.line(screen_coords[idx0], screen_coords[idx1], radius=2, color=0x00FFFF)
        
        # 显示提示文字
        gui.text(f"插值进度: {t:.2f}", pos=(0.05, 0.95), color=0xffffff)
        gui.text(f"空格切换 | 退出=ESC", pos=(0.05, 0.9), color=0xffffff)
        gui.show()

if __name__ == '__main__':
    main()