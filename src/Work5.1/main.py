import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3dd.utils import ico_sphere
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader
)

# ===================== 1. 设备配置 =====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== 2. 可视化函数 =====================
def visualize_silhouette(current, target, epoch):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(target, cmap="gray")
    plt.title("Target Silhouette")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(current, cmap="gray")
    plt.title(f"Optimized Silhouette Epoch {epoch}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ===================== 3. 正则化损失函数 =====================
def laplacian_smoothing_loss(mesh):
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    e0, e1 = edges[:, 0], edges[:, 1]
    sum_neigh = torch.zeros_like(verts).scatter_add(0, e0.unsqueeze(1).repeat(1,3), verts[e1])
    cnt = torch.zeros(verts.shape[0], 1, device=device).scatter_add(0, e0.unsqueeze(1), torch.ones_like(e0.unsqueeze(1)))
    mean_neigh = sum_neigh / cnt.clamp(min=1)
    return torch.mean((verts - mean_neigh) ** 2)

def edge_length_penalty_loss(mesh):
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    len_edges = torch.norm(verts[edges[:,0]] - verts[edges[:,1]], dim=1)
    mean_len = len_edges.mean()
    return torch.mean((len_edges - mean_len) ** 2)

def normal_consistency_loss(mesh):
    normals = mesh.faces_normals_packed()
    face_adj = mesh.face_edges_packed()
    n0, n1 = normals[face_adj[:,0]], normals[face_adj[:,1]]
    cos_sim = torch.sum(n0 * n1, dim=1)
    return torch.mean((1 - cos_sim) ** 2)

# ===================== 4. 加载目标模型 & 构建渲染器 =====================
# 加载奶牛目标网格
target_mesh = load_objs_as_meshes(["cow.obj"], device=device)

# 多视角相机
num_views = 6
angles = torch.linspace(0, 2*np.pi, num_views)
R = []
T = []
for angle in angles:
    rot = torch.tensor([
        [torch.cos(angle), 0, torch.sin(angle)],
        [0, 1, 0],
        [-torch.sin(angle), 0, torch.cos(angle)]
    ]).unsqueeze(0)
    trans = torch.tensor([[0, 0, 3.0]])
    R.append(rot)
    T.append(trans)
R = torch.cat(R, dim=0).to(device)
T = torch.cat(T, dim=0).to(device)

cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 软光栅化配置
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.005,
    faces_per_pixel=50,
)

# 剪影渲染器
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader()
)

# 生成目标剪影
with torch.no_grad():
    target_sil = renderer(target_mesh)[..., 3]  # alpha通道作为剪影

# ===================== 5. 初始化球体网格 & 可微偏移 =====================
src_mesh = ico_sphere(level=5, device=device)
deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True, device=device)

# ===================== 6. 优化参数设置 =====================
optimizer = torch.optim.Adam([deform_verts], lr=1e-3)
epochs = 300
# 正则权重
w_lap = 10.0
w_edge = 1.0
w_normal = 5.0

# ===================== 7. 梯度下降优化循环 =====================
for epoch in range(epochs):
    optimizer.zero_grad()

    # 形变网格
    def_mesh = src_mesh.offset_verts(deform_verts)

    # 渲染当前剪影
    curr_sil = renderer(def_mesh)[..., 3]

    # 剪影匹配损失
    loss_sil = torch.mean((curr_sil - target_sil) ** 2)

    # 正则损失
    loss_lap = laplacian_smoothing_loss(def_mesh)
    loss_edge = edge_length_penalty_loss(def_mesh)
    loss_norm = normal_consistency_loss(def_mesh)

    # 总损失
    total_loss = loss_sil + w_lap * loss_lap + w_edge * loss_edge + w_normal * loss_norm

    # 反向传播更新
    total_loss.backward()
    optimizer.step()

    # 打印日志
    if (epoch + 1) % 10 == 0:
        print(f"Epoch:{epoch+1:3d} | TotalLoss:{total_loss.item():.6f} | SilLoss:{loss_sil.item():.6f}")

    # 可视化中间结果
    if (epoch + 1) % 50 == 0:
        visualize_silhouette(
            curr_sil[0].detach().cpu().numpy(),
            target_sil[0].detach().cpu().numpy(),
            epoch+1
        )

# ===================== 8. 保存最终优化模型 =====================
final_mesh = src_mesh.offset_verts(deform_verts)
save_obj("optimized_cow_mesh.obj", final_mesh.verts_packed(), final_mesh.faces_packed())
print("优化完成，模型已保存为 optimized_cow_mesh.obj")