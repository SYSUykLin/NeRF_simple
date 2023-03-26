'''
1.Predict occupancy
2.Perform marching cube algorithm
    前面两个得到mesh
3.Remove noise
4.Compute color for each vertex
    后面一个是上色
'''
import numpy as np
import torch
import mcubes
import trimesh
from tqdm import tqdm
import models
import tools


def tights_bounds(bounding_box, N=128, N_rays=1024, sigma_threshold=20):
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    zmin, zmax = -2.64, -0.64
    sigma_threshold = 20.0

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    print('Predicting occupancy ...')

    coordinate_embeddings = models.hash_embedding(bounding_box)
    direction_embeddings = models.SHEncoder()

    coordinate_embeddings.load_state_dict(torch.load('models\\coordinate_embeddings.pth'))
    direction_embeddings.load_state_dict(torch.load('models\\direction_embeddings.pth'))
    fine_model = models.nerf_ngp(coordinate_embeddings.count_dim(), direction_embeddings.count_dim())
    fine_model.load_state_dict(torch.load('models\\fine_model.pth'))
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
    mesh = trimesh.Trimesh(vertices/N,  triangles)
    mesh.show()

datadir = ".\\dataset\\nerf_synthetic\\ship"
data_config = 'dataset\\ship.txt'
return_datasets, (H, W, focal), render_poses, min_bound, max_bound = tools.read_datasets(datadir, data_config)
tights_bounds((min_bound, max_bound))
    
