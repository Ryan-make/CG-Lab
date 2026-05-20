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
sample_count = ti.field(ti.i32, shape=())  # 抗锯齿采样数

# 初始值设置
@ti.kernel
def init_params():
    light_pos[None] = tm.vec3(2.0, 4.0, 3.0)
    max_bounces[None] = 6  # 玻璃需要更多反弹
    sample_count[None] = 4 # MSAA 4x采样

init_params()

# ===================== 材质ID定义 =====================
MAT_NONE = 0       # 无相交
MAT_GROUND = 1     # 地面（漫反射+棋盘格）
MAT_GLASS_SPHERE = 2 # 玻璃球（新增）
MAT_SILVER_SPHERE =3# 银色镜面球

# ===================== 光线求交核心数据结构 =====================
@ti.dataclass
class HitInfo:
    hit: ti.i32       # 是否相交 1=相交 0=未相交
    t: ti.f32         # 光线方程参数 t
    pos: tm.vec3      # 相交点坐标
    normal: tm.vec3   # 相交点法线
    mat_id: ti.i32    # 材质ID
    inside: ti.i32    # 是否在物体内部（玻璃专用）

# ===================== 场景几何体隐式定义 =====================
@ti.func
def ray_cast(ray_origin: tm.vec3, ray_dir: tm.vec3) -> HitInfo:
    info = HitInfo(0, 1e8, tm.vec3(0), tm.vec3(0), MAT_NONE, 0)
    eps = 1e-4

    # 1. 地面
    if abs(ray_dir.y) > eps:
        t_plane = (-1.0 - ray_origin.y) / ray_dir.y
        if t_plane > eps and t_plane < info.t:
            info.hit = 1
            info.t = t_plane
            info.pos = ray_origin + t_plane * ray_dir
            info.normal = tm.vec3(0, 1, 0)
            info.mat_id = MAT_GROUND

    # 2. 玻璃球（原红球替换）
    sphere1_center = tm.vec3(-1.0, 0.0, 0.0)
    oc = ray_origin - sphere1_center
    a = tm.dot(ray_dir, ray_dir)
    b = 2.0 * tm.dot(oc, ray_dir)
    c = tm.dot(oc, oc) - 1.0
    disc = b*b - 4*a*c
    if disc > 0.0:
        t1 = (-b - ti.sqrt(disc)) / (2*a)
        t2 = (-b + ti.sqrt(disc)) / (2*a)
        t = t1
        inside = 0
        if t1 < eps:
            t = t2
            inside = 1
        if t > eps and t < info.t:
            info.hit = 1
            info.t = t
            info.pos = ray_origin + t * ray_dir
            info.normal = tm.normalize(info.pos - sphere1_center)
            info.mat_id = MAT_GLASS_SPHERE
            info.inside = inside
            if inside:
                info.normal = -info.normal

    # 3. 银色镜面球
    sphere2_center = tm.vec3(1.0, 0.0, 0.0)
    oc2 = ray_origin - sphere2_center
    a2 = tm.dot(ray_dir, ray_dir)
    b2 = 2.0 * tm.dot(oc2, ray_dir)
    c2 = tm.dot(oc2, oc2) - 1.0
    disc2 = b2*b2 - 4*a2*c2
    if disc2 > 0.0:
        t2 = (-b2 - ti.sqrt(disc2)) / (2*a2)
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
        scale = 2.0
        x = ti.floor(hit.pos.x * scale)
        z = ti.floor(hit.pos.z * scale)
        f = (int(x) + int(z)) % 2
        color = tm.vec3(1.0) if f == 0 else tm.vec3(0.05)

    result = tm.vec3(0.0) if shadow else color * ndotl
    return result

# ===================== 玻璃折射（斯涅尔定律 + 全反射）=====================
@ti.func
def refract_ray(ray_dir: tm.vec3, normal: tm.vec3, inside: ti.i32) -> tm.vec3:
    # 空气折射率 ~1.0，玻璃折射率 ~1.5
    n1 = 1.0
    n2 = 1.5
    if inside:
        n1, n2 = n2, n1

    eta = n1 / n2
    cos_i = tm.dot(-ray_dir, normal)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)

    refracted = tm.vec3(0)
    # 全反射判断
    if sin2_t < 1.0:
        cos_t = ti.sqrt(1.0 - sin2_t)
        refracted = eta * ray_dir + (eta * cos_i - cos_t) * normal
    else:
        # 全反射
        refracted = tm.reflect(ray_dir, normal)
    return tm.normalize(refracted)

# ===================== 单像素光线追踪 =====================
@ti.func
def render_pixel(u: ti.f32, v: ti.f32) -> tm.vec3:
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
            final_color = tm.vec3(0.02, 0.05, 0.1) * throughput
            break

        # 镜面反射
        if hit.mat_id == MAT_SILVER_SPHERE:
            ray_d = tm.reflect(ray_d, hit.normal)
            ray_o = hit.pos + hit.normal * eps
            throughput *= 0.95

        # 玻璃折射（斯涅尔定律）
        elif hit.mat_id == MAT_GLASS_SPHERE:
            ray_d = refract_ray(ray_d, hit.normal, hit.inside)
            ray_o = hit.pos + ray_d * eps
            throughput *= 0.98

        # 漫反射
        else:
            final_color = shade_diffuse(hit) * throughput
            break

    return final_color

# ===================== 渲染内核（MSAA 抗锯齿）=====================
@ti.kernel
def render():
    for i, j in pixels:
        color = tm.vec3(0.0)
        # 多采样抗锯齿：每个像素多次采样取平均
        for _ in range(sample_count[None]):
            u = (i + ti.random()) / res[0]
            v = (j + ti.random()) / res[1]
            color += render_pixel(u, v)
        pixels[i, j] = tm.clamp(color / sample_count[None], 0.0, 1.0)

# ===================== UI窗口与交互 =====================
window = ti.ui.Window("Ray Tracing: Glass + MSAA", res=res)
canvas = window.get_canvas()
gui = window.get_gui()

# 主循环
while window.running:
    render()
    canvas.set_image(pixels)

    with gui.sub_window("Controls", 0.7, 0.1, 0.25, 0.35):
        light_x = gui.slider_float("Light X", light_pos[None].x, -5, 5)
        light_y = gui.slider_float("Light Y", light_pos[None].y, 0, 5)
        light_z = gui.slider_float("Light Z", light_pos[None].z, -5, 5)
        light_pos[None] = tm.vec3(light_x, light_y, light_z)

        bounce_val = gui.slider_int("Max Bounces", max_bounces[None], 1, 8)
        max_bounces[None] = bounce_val

        aa_val = gui.slider_int("MSAA Samples", sample_count[None], 1, 8)
        sample_count[None] = aa_val

    window.show()