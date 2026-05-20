import taichi as ti
import taichi.ui as ui

# 初始化Taichi
ti.init(arch=ti.gpu)

# 画布分辨率
res_x = 800
res_y = 600

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# ===================== 光照材质参数 =====================
ka = ti.field(ti.f32, shape=())
kd = ti.field(ti.f32, shape=())
ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

@ti.kernel
def init_params():
    ka[None] = 0.2
    kd[None] = 0.7
    ks[None] = 0.5
    shininess[None] = 32.0

# ===================== 固定场景参数 =====================
camera_pos = ti.Vector([0.0, 0.0, 5.0])
light_pos = ti.Vector([2.0, 3.0, 4.0])
light_color = ti.Vector([1.0, 1.0, 1.0])
bg_color = ti.Vector([0.0, 0.2, 0.3])

# 球体
sphere_center = ti.Vector([-1.2, -0.2, 0.0])
sphere_radius = 1.2
sphere_color = ti.Vector([0.8, 0.1, 0.1])

# 圆锥
cone_tip = ti.Vector([1.2, 1.2, 0.0])
cone_base_y = -1.4
cone_radius = 1.2
cone_color = ti.Vector([0.6, 0.2, 0.8])

# ===================== 光线-物体求交（通用函数） =====================
@ti.func
def ray_sphere_intersect(ray_origin, ray_dir, center, radius):
    oc = ray_origin - center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4 * a * c
    t = -1.0
    if discriminant >= 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
    return t > 0, t

@ti.func
def ray_cone_intersect(ray_origin, ray_dir, tip, base_y, radius):
    height = tip.y - base_y
    tan_theta = radius / height
    tan_sq = tan_theta * tan_theta

    ox, oy, oz = ray_origin
    dx, dy, dz = ray_dir
    tx, ty, tz = tip

    a = dx*dx + dz*dz - tan_sq * dy*dy
    b = 2 * ((ox - tx)*dx + (oz - tz)*dz + tan_sq * (ty - oy)*dy)
    c = (ox - tx) * (ox - tx) + (oz - tz) * (oz - tz) - tan_sq * (ty - oy) * (ty - oy)

    discriminant = b*b - 4*a*c
    t = -1.0
    if discriminant >= 0 and a != 0:
        t0 = (-b - ti.sqrt(discriminant)) / (2*a)
        y_hit = oy + t0 * dy
        if t0 > 0 and y_hit <= tip.y and y_hit >= base_y:
            t = t0
    return t > 0, t

@ti.func
def get_sphere_normal(hit_pos, center):
    return (hit_pos - center).normalized()

@ti.func
def get_cone_normal(hit_pos, tip, base_y, radius):
    height = tip.y - base_y
    tan_theta = radius / height
    x, y, z = hit_pos
    tx, ty, tz = tip
    nx = x - tx
    ny = (ty - y) * tan_theta * tan_theta
    nz = z - tz
    return ti.Vector([nx, ny, nz]).normalized()

# ===================== 【新增】硬阴影判断：Shadow Ray =====================
@ti.func
def is_in_shadow(shadow_origin):
    """
    从交点发射阴影射线到光源，判断是否被遮挡
    返回 True = 在阴影中，False = 被直射
    """
    # 阴影射线方向：指向光源
    shadow_dir = (light_pos - shadow_origin).normalized()
    # 光源距离（防止误判自身相交）
    light_dist = (light_pos - shadow_origin).norm()

    # 偏移起点，避免浮点误差导致自身相交（解决黑噪点）
    shadow_origin_biased = shadow_origin + shadow_dir * 0.001

    # 检测与球体、圆锥是否相交
    hit_sphere, t_sphere = ray_sphere_intersect(shadow_origin_biased, shadow_dir, sphere_center, sphere_radius)
    hit_cone, t_cone = ray_cone_intersect(shadow_origin_biased, shadow_dir, cone_tip, cone_base_y, cone_radius)

    in_shadow = False
    # 相交点在光源之前 → 阴影
    if (hit_sphere and t_sphere < light_dist) or (hit_cone and t_cone < light_dist):
        in_shadow = True
    return in_shadow

# ===================== 【升级】Blinn-Phong 光照 =====================
@ti.func
def blinn_phong_shading(hit_pos, normal, obj_color, in_shadow):
    # 单位向量（必须归一化！）
    light_dir = (light_pos - hit_pos).normalized()
    view_dir = (camera_pos - hit_pos).normalized()
    # Blinn-Phong 核心：半程向量 H
    half_dir = (light_dir + view_dir).normalized()

    # 环境光（阴影中只保留环境光）
    ambient = ka[None] * light_color * obj_color

    # 漫反射
    diff = ti.max(0.0, normal.dot(light_dir))
    diffuse = kd[None] * diff * light_color * obj_color

    # Blinn-Phong 高光：使用 N·H 替代 R·V
    spec = ti.pow(ti.max(0.0, normal.dot(half_dir)), shininess[None])
    specular = ks[None] * spec * light_color

    # 总颜色 + 截断防过曝
    total = ambient + diffuse + specular
    
    # 根据是否在阴影中选择最终颜色
    result = ambient if in_shadow else total
    return ti.math.clamp(result, 0.0, 1.0)

# ===================== 核心渲染 =====================
@ti.kernel
def render():
    for i, j in pixels:
        u = (i / res_x) * 2 - 1
        v = (j / res_y) * 2 - 1
        u *= res_x / res_y
        ray_dir = ti.Vector([u, v, -1.0]).normalized()

        # 求交
        hit_sphere, t_sphere = ray_sphere_intersect(camera_pos, ray_dir, sphere_center, sphere_radius)
        hit_cone, t_cone = ray_cone_intersect(camera_pos, ray_dir, cone_tip, cone_base_y, cone_radius)

        # 深度测试（最近物体）
        min_t = -1.0
        hit_obj = 0
        if hit_sphere and hit_cone:
            min_t = t_sphere if t_sphere < t_cone else t_cone
            hit_obj = 1 if t_sphere < t_cone else 2
        elif hit_sphere:
            min_t = t_sphere
            hit_obj = 1
        elif hit_cone:
            min_t = t_cone
            hit_obj = 2

        # 着色
        if hit_obj != 0:
            hit_pos = camera_pos + min_t * ray_dir
            normal = ti.Vector([0.0, 0.0, 0.0])
            obj_color = ti.Vector([0.0, 0.0, 0.0])

            if hit_obj == 1:
                normal = get_sphere_normal(hit_pos, sphere_center)
                obj_color = sphere_color
            else:
                normal = get_cone_normal(hit_pos, cone_tip, cone_base_y, cone_radius)
                obj_color = cone_color

            # 阴影判断 + Blinn-Phong着色
            in_shadow = is_in_shadow(hit_pos)
            pixels[i, j] = blinn_phong_shading(hit_pos, normal, obj_color, in_shadow)
        else:
            pixels[i, j] = bg_color

# ===================== UI 主循环 =====================
def main():
    init_params()
    window = ui.Window("Blinn-Phong + Hard Shadow 实验", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        ka[None] = gui.slider_float("Ka(环境光)", ka[None], 0.0, 1.0)
        kd[None] = gui.slider_float("Kd(漫反射)", kd[None], 0.0, 1.0)
        ks[None] = gui.slider_float("Ks(高光)", ks[None], 0.0, 1.0)
        shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)

        render()
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()