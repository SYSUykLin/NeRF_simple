# pyglet要安装版本小于1的
import numpy as np
import torch
import mcubes
import trimesh
from tqdm import tqdm
import models
import tools
import math
# 这包有点难装，要markupsafe-2.0.1，werkzeug-2.0.3这些降级的包
import open3d as o3d
from plyfile import PlyData, PlyElement

# 解决mcubes.marching_cubes出现的版本冲突问题，我也不知道为啥能解决，防止就行了
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def find_min_max_xyz(bounding_box):
    x_min, y_min, z_min = math.inf, math.inf, math.inf
    x_max, y_max, z_max = -math.inf, -math.inf, -math.inf
    min_bound, max_bound = bounding_box
    if x_min > min_bound[0]:
        x_min = min_bound[0]
    if x_max < max_bound[0]:
        x_max = max_bound[0]
    if y_min > min_bound[1]:
        y_min = min_bound[1]
    if y_max < max_bound[1]:
        y_max = max_bound[1]
    if z_min > min_bound[2]:
        z_min = min_bound[2]
    if z_max < max_bound[2]:
        z_max = max_bound[2]
    return x_min - 1, x_max + 1, y_min - 1, y_max + 1, z_min - 1, z_max + 1
    

def tights_bounds(bounding_box, N=128, N_rays=1024, 
                  sigma_threshold=20, 
                  models_dir="D:\\NeRF\\NeRF_project\\models", device='cuda'):
    
    xmin, xmax, ymin, ymax, zmin, zmax = find_min_max_xyz(bounding_box)
    sigma_threshold = 20.0

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    print('Predicting occupancy ...')
    
    coords_embeddings_dir = models_dir + "\\coordinate_embeddings.pt"
    coordinate_embeddings = torch.load(coords_embeddings_dir).to(device)
    direction_embeddings_dir = models_dir + "\\direction_embeddings.pt"
    direction_embeddings = torch.load(direction_embeddings_dir).to(device)
    fine_model = models.nerf_ngp(coordinate_embeddings.count_dim(), direction_embeddings.count_dim())
    fine_model_dir = models_dir + "\\fine_model.pt"
    fine_model = torch.load(fine_model_dir).to(device)
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, N_rays)):
            xyz_embedded, keep_mask = coordinate_embeddings(xyz_[i:i+N_rays]) # (N, embed_xyz_channels)
            dir_embedded = direction_embeddings(dir_[i:i+N_rays]) # (N, embed_dir_channels)
            out_chunks += [fine_model(xyz_embedded, dir_embedded, keep_mask)]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    print('Extracting  Mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)

    
    mesh = trimesh.Trimesh(vertices / N, triangles)
    mesh.show()

datadir = "D:\\NeRF\\NeRF_project\\dataset\\nerf_synthetic\\lego"
data_config = 'dataset\\lego.txt'
return_datasets, (H, W, focal), render_poses, min_bound, max_bound = tools.read_datasets(datadir, data_config)
models_dir = "D:\\NeRF\\NeRF_project\\models"
tights_bounds((min_bound, max_bound), models_dir=models_dir)