import tools
import models
import torch
import random
import render


def train_nerf(datadir, dataconfig, gpu=True):
    print(f"use GPU : {gpu}")
    datasets, (H, W, focal), render_poses = tools.read_datasets(datadir, 
                                                                dataconfig)
    print(f"images shape : {H, W}")

    '''
    1.创建NeRF
    2.渲染
    3.训练
    '''

    epochs = 200000
    test_iter = 100
    coordinate_L = 10
    direction_L = 4
    N_rays = 4096
    N_samples = 64
    N_importances = 128
    near = 2.
    far = 6.
    train_data = datasets['train']
    val_data = datasets['validate']
    test_data = datasets['test']

    coordinate_embeddings = models.fourier_embedding(L=coordinate_L)
    direction_embeddings = models.fourier_embedding(L=direction_L)
    coord_input_dim = coordinate_embeddings.count_dim()
    direct_input_dim = direction_embeddings.count_dim()

    coarse_model = models.nerf(coord_input_dim, direct_input_dim, 
                               coordinate_embeddings, direction_embeddings)
    find_model = models.nerf(coord_input_dim, direct_input_dim, 
                             coordinate_embeddings, direction_embeddings)
    
    train_images = train_data["images"]
    train_poses = train_data["poses"]
    N_images, Height, Width, channal = train_images.shape
    # tensor和Tensor不一样，Tensor是生成单精度，tensor是复制之前的精度
    K = torch.Tensor([
                     [focal, 0, 0.5 * W],
                     [0, focal, 0.5 * H],
                     [0, 0, 1]
                     ])
    
    if gpu:
        K = K.cuda()
        coarse_model = coarse_model.cuda()
        find_model = find_model.cuda()

    for i in range(epochs):
        # 一张图片一张图片训练
        index_image = random.sample(range(N_images), 1)
        train_image = train_images[index_image]
        train_pose = train_poses[index_image]

        if gpu:
            train_image = train_image.cuda()
            train_pose = train_pose.cuda()

        rays_d, rays_o = render.get_rays(H, W, K, train_pose, gpu)
        grid_W, grid_H = torch.meshgrid(torch.arange(W), torch.arange(H), 
                                        indexing="xy")
        grid = torch.stack((grid_H, grid_W), dim=-1)
        grid = grid.reshape((-1, 2))
        select_indexs = random.sample(range(grid.shape[0]), N_rays)
        select_coords = grid[select_indexs]
        
        train_image = train_image.reshape(Height, Width, channal)
        select_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1], :]      
        select_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1], :]
        targets_image = train_image[select_coords[:, 0], select_coords[:, 1]]
       
        z_vals = render.coarse_samples(N_rays, near, far, N_samples, True)
        raw, viewdirs, coordinates = render.generate_raw(z_vals, 
                                                         coarse_model, 
                                                         select_rays_d, 
                                                         select_rays_o)
        # viewdirs：归一化后的方向
        color = raw[..., :3]
        sigma = raw[..., -1]



        import sys
        sys.exit()
        
        



    


    
    


