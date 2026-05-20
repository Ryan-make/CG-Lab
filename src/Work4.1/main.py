import taichi as ti
import taichi.math as tm

# 初始化Taichi，使用GPU加速
ti.init(arch=ti.gpu)

# 渲染分辨率设置
res = (800, 600)
# 像素缓冲区，存储最终渲染颜色
pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

# ===================== 可交互参数域 =====================
light_pos = ti.Vector.field(3, ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 初始值设置（和示例视频里的参数一致）
@ti.kernel
def init_params():
    light_pos[None] = tm.vec3(2.0, 4.0, 3.0)
    max_bounces[None] = 3

init_params()

# ===================== 材质ID定义 =====================
MAT_NONE = 0       # 无相交
MAT_GROUND = 1     # 地面（漫反射+棋盘格）
MAT_RED_SPHERE = 2 # 红色漫反射球
MAT_SILVER_SPHERE =3# 银色镜面球

# ===================== 光线求交核心数据结构 =====================
@ti.dataclass
class HitInfo:
    hit: ti.i32       # 是否相交 1=相交 0=未相交
    t: ti.f32         # 光线方程参数 t
    pos: tm.vec3      # 相交点坐标
    normal: tm.vec3   # 相交点法线
    mat_id: ti.i32    # 材质ID

# ===================== 场景几何体隐式定义 =====================
@ti.func
def ray_cast(ray_origin: tm.vec3, ray_dir: tm.vec3) -> HitInfo:
    # 初始化：未相交
    info = HitInfo(0, 1e8, tm.vec3(0), tm.vec3(0), MAT_NONE)
    # 浮点精度最小值
    eps = 1e-4

    # ---------------- 1. 无限大地面 y=-1，法线(0,1,0) ----------------
    if abs(ray_dir.y) > eps:
        t_plane = (-1.0 - ray_origin.y) / ray_dir.y
        if t_plane > eps and t_plane < info.t:
            info.hit = 1
            info.t = t_plane
            info.pos = ray_origin + t_plane * ray_dir
            info.normal = tm.vec3(0, 1, 0)
            info.mat_id = MAT_GROUND

    # ---------------- 2. 红色漫反射球：中心(-1.0, 0.0, 0.0)，半径1 ----------------
    sphere1_center = tm.vec3(-1.0, 0.0, 0.0)
    oc = ray_origin - sphere1_center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - 1.0
    disc = b*b - 4*a*c
    if disc > 0.0:
        t1 = (-b - ti.sqrt(disc)) / (2.0*a)
        if t1 > eps and t1 < info.t:
            info.hit = 1
            info.t = t1
            info.pos = ray_origin + t1 * ray_dir
            info.normal = tm.normalize(info.pos - sphere1_center)
            info.mat_id = MAT_RED_SPHERE

    # ---------------- 3. 银色镜面球：中心(1.0, 0.0, 0.0)，半径1 ----------------
    sphere2_center = tm.vec3(1.0, 0.0, 0.0)
    oc2 = ray_origin - sphere2_center
    a2 = tm.dot(ray_dir, ray_dir)
    b2 = 2.0 * tm.dot(oc2, ray_dir)
    c2 = tm.dot(oc2, oc2) - 1.0
    disc2 = b2*b2 - 4*a2*c2
    if disc2 > 0.0:
        t2 = (-b2 - ti.sqrt(disc2)) / (2.0*a2)
        if t2 > eps and t2 < info.t:
            info.hit = 1
            info.t = t2
            info.pos = ray_origin + t2 * ray_dir
            info.normal = tm.normalize(info.pos - sphere2_center)
            info.mat_id = MAT_SILVER_SPHERE

    return info

# ===================== 硬阴影判断 =====================
@ti.func
def is_shadowed(p: tm.vec3, normal: tm.vec3) -> ti.i32:
    eps = 1e-4
    light_dir = light_pos[None] - p
    light_dist = tm.length(light_dir)
    shadow_ray_dir = tm.normalize(light_dir)
    # 沿法线外偏，避免自相交
    shadow_origin = p + normal * eps
    hit = ray_cast(shadow_origin, shadow_ray_dir)
    return 1 if (hit.hit and hit.t < light_dist - eps) else 0

# ===================== 漫反射着色 =====================
@ti.func
def shade_diffuse(hit: HitInfo) -> tm.vec3:
    shadow = is_shadowed(hit.pos, hit.normal)

    light_dir = tm.normalize(light_pos[None] - hit.pos)
    ndotl = max(tm.dot(hit.normal, light_dir), 0.0)

    color = tm.vec3(0.0)
    if hit.mat_id == MAT_GROUND:
        # 棋盘格（和示例一致的黑白格子）
        scale = 2.0
        x = ti.floor(hit.pos.x * scale)
        z = ti.floor(hit.pos.z * scale)
        f = (int(x) + int(z)) % 2
        color = tm.vec3(1.0) if f == 0 else tm.vec3(0.05)
    elif hit.mat_id == MAT_RED_SPHERE:
        color = tm.vec3(1.0, 0.15, 0.15)

    result = tm.vec3(0.0) if shadow else color * ndotl
    return result

# ===================== 单像素光线追踪 =====================
@ti.func
def render_pixel(u: ti.f32, v: ti.f32) -> tm.vec3:
    # 相机设置：和示例视角一致
    cam_pos = tm.vec3(0.0, 1.0, 6.0)
    uv = tm.vec2(u, v) * 2.0 - 1.0
    uv.x *= res[0] / res[1]
    ray_dir = tm.normalize(tm.vec3(uv.x, uv.y, -1.5))

    throughput = 1.0
    final_color = tm.vec3(0.0)
    ray_o = cam_pos
    ray_d = ray_dir
    eps = 1e-4

    for _ in range(max_bounces[None]):
        hit = ray_cast(ray_o, ray_d)
        if not hit.hit:
            # 背景色：深黑色（和示例一致）
            final_color = tm.vec3(0.02, 0.05, 0.1) * throughput
            break

        if hit.mat_id == MAT_SILVER_SPHERE:
            # 镜面反射
            ray_d = tm.reflect(ray_d, hit.normal)
            ray_o = hit.pos + hit.normal * eps
            throughput *= 0.95
        else:
            final_color = shade_diffuse(hit) * throughput
            break

    return final_color

# ===================== 渲染内核 =====================
@ti.kernel
def render():
    for i, j in pixels:
        u = (i + ti.random()) / res[0]
        v = (j + ti.random()) / res[1]
        color = render_pixel(u, v)
        pixels[i, j] = tm.clamp(color, 0.0, 1.0)

# ===================== UI窗口与交互 =====================
window = ti.ui.Window("Ray Tracing Demo", res=res)
canvas = window.get_canvas()
gui = window.get_gui()

# 主循环
while window.running:
    render()
    canvas.set_image(pixels)

    # 和示例一致的控制面板
    with gui.sub_window("Controls", 0.7, 0.1, 0.25, 0.3):
        light_x = gui.slider_float("Light X", light_pos[None].x, -5, 5)
        light_y = gui.slider_float("Light Y", light_pos[None].y, 0, 5)
        light_z = gui.slider_float("Light Z", light_pos[None].z, -5, 5)
        light_pos[None] = tm.vec3(light_x, light_y, light_z)

        bounce_val = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
        max_bounces[None] = bounce_val

    window.show()