import taichi as ti
import taichi.ui as ui

# 初始化Taichi（GPU渲染，速度更快）
ti.init(arch=ti.gpu)

# 画布分辨率
res_x = 800
res_y = 600

# 像素缓冲区：存储每个像素的RGB颜色
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# ===================== 光照材质参数（可通过UI实时调节） =====================
# 环境光系数
ka = ti.field(ti.f32, shape=())
# 漫反射系数
kd = ti.field(ti.f32, shape=())
# 镜面高光系数
ks = ti.field(ti.f32, shape=())
# 高光指数
shininess = ti.field(ti.f32, shape=())

# 初始化默认参数
@ti.kernel
def init_params():
    ka[None] = 0.2
    kd[None] = 0.7
    ks[None] = 0.5
    shininess[None] = 32.0

# ===================== 固定场景参数 =====================
# 摄像机位置
camera_pos = ti.Vector([0.0, 0.0, 5.0])
# 光源位置
light_pos = ti.Vector([2.0, 3.0, 4.0])
# 光源颜色（白色）
light_color = ti.Vector([1.0, 1.0, 1.0])
# 背景颜色（深青色）
bg_color = ti.Vector([0.0, 0.2, 0.3])

# 球体参数
sphere_center = ti.Vector([-1.2, -0.2, 0.0])
sphere_radius = 1.2
sphere_color = ti.Vector([0.8, 0.1, 0.1])  # 深红色

# 圆锥参数
cone_tip = ti.Vector([1.2, 1.2, 0.0])      # 圆锥顶点
cone_base_y = -1.4                        # 圆锥底面高度
cone_radius = 1.2                         # 底面半径
cone_color = ti.Vector([0.6, 0.2, 0.8])    # 紫色

# ===================== 光线-物体求交函数 =====================
@ti.func
def ray_sphere_intersect(ray_origin, ray_dir, center, radius):
    """
    光线-球体求交
    返回：(是否相交, 交点距离t)
    """
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
    """
    光线-圆锥求交（有限高度圆锥）
    返回：(是否相交, 交点距离t)
    """
    # 圆锥高度与斜率
    height = tip.y - base_y
    tan_theta = radius / height
    tan_sq = tan_theta * tan_theta

    ox, oy, oz = ray_origin
    dx, dy, dz = ray_dir
    tx, ty, tz = tip

    # 圆锥求交公式
    a = dx*dx + dz*dz - tan_sq * dy*dy
    b = 2 * ((ox - tx)*dx + (oz - tz)*dz + tan_sq * (ty - oy)*dy)
    c = (ox - tx) * (ox - tx) + (oz - tz) * (oz - tz) - tan_sq * (ty - oy) * (ty - oy)

    discriminant = b*b - 4*a*c
    t = -1.0
    if discriminant >= 0 and a != 0:
        t0 = (-b - ti.sqrt(discriminant)) / (2*a)
        y_hit = oy + t0 * dy
        # 限制交点在圆锥顶点和底面之间
        if t0 > 0 and y_hit <= tip.y and y_hit >= base_y:
            t = t0
    return t > 0, t

@ti.func
def get_sphere_normal(hit_pos, center):
    """计算球体法向量"""
    return (hit_pos - center).normalized()

@ti.func
def get_cone_normal(hit_pos, tip, base_y, radius):
    """计算圆锥法向量"""
    height = tip.y - base_y
    tan_theta = radius / height
    x, y, z = hit_pos
    tx, ty, tz = tip

    nx = x - tx
    ny = (ty - y) * tan_theta * tan_theta
    nz = z - tz
    return ti.Vector([nx, ny, nz]).normalized()

# ===================== Phong 光照计算 =====================
@ti.func
def phong_shading(hit_pos, normal, obj_color):
    """
    Phong光照模型：环境光+漫反射+镜面高光
    """
    # 1. 计算方向向量
    light_dir = (light_pos - hit_pos).normalized()  # L：指向光源
    view_dir = (camera_pos - hit_pos).normalized()   # V：指向摄像机
    reflect_dir = 2 * normal.dot(light_dir) * normal - light_dir  # R：反射向量

    # 2. 环境光
    ambient = ka[None] * light_color * obj_color

    # 3. 漫反射（Lambert定律）
    diff = ti.max(0.0, normal.dot(light_dir))
    diffuse = kd[None] * diff * light_color * obj_color

    # 4. 镜面高光
    spec = ti.pow(ti.max(0.0, reflect_dir.dot(view_dir)), shininess[None])
    specular = ks[None] * spec * light_color

    # 总光照
    return ambient + diffuse + specular

# ===================== 核心渲染内核 =====================
@ti.kernel
def render():
    for i, j in pixels:
        # 屏幕坐标归一化（-1~1）
        u = (i / res_x) * 2 - 1
        v = (j / res_y) * 2 - 1
        # 适配宽高比
        u *= res_x / res_y

        # 构建光线：起点=摄像机，方向=屏幕点-摄像机
        ray_dir = ti.Vector([u, v, -1.0]).normalized()

        # 光线与两个物体求交
        hit_sphere, t_sphere = ray_sphere_intersect(camera_pos, ray_dir, sphere_center, sphere_radius)
        hit_cone, t_cone = ray_cone_intersect(camera_pos, ray_dir, cone_tip, cone_base_y, cone_radius)

        # 深度测试：选择最近的交点（最小正t）
        min_t = -1.0
        hit_obj = 0  # 0=无碰撞，1=球体，2=圆锥
        if hit_sphere and hit_cone:
            if t_sphere < t_cone:
                min_t = t_sphere
                hit_obj = 1
            else:
                min_t = t_cone
                hit_obj = 2
        elif hit_sphere:
            min_t = t_sphere
            hit_obj = 1
        elif hit_cone:
            min_t = t_cone
            hit_obj = 2

        # 着色
        if hit_obj != 0:
            # 计算交点坐标
            hit_pos = camera_pos + min_t * ray_dir
            # 计算法向量
            normal = ti.Vector([0.0, 0.0, 0.0])
            obj_color = ti.Vector([0.0, 0.0, 0.0])
            if hit_obj == 1:
                normal = get_sphere_normal(hit_pos, sphere_center)
                obj_color = sphere_color
            else:
                normal = get_cone_normal(hit_pos, cone_tip, cone_base_y, cone_radius)
                obj_color = cone_color

            # Phong着色
            pixels[i, j] = phong_shading(hit_pos, normal, obj_color)
        else:
            # 无碰撞：背景色
            pixels[i, j] = bg_color

# ===================== UI交互与主循环 =====================
def main():
    init_params()
    # 创建窗口
    window = ui.Window("Phong光照模型实验", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        # 绘制UI滑动条
        ka[None] = gui.slider_float("Ka(环境光系数)", ka[None], 0.0, 1.0)
        kd[None] = gui.slider_float("Kd(漫反射系数)", kd[None], 0.0, 1.0)
        ks[None] = gui.slider_float("Ks(高光系数)", ks[None], 0.0, 1.0)
        shininess[None] = gui.slider_float("Shininess(高光指数)", shininess[None], 1.0, 128.0)

        # 渲染
        render()

        # 显示图像
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()