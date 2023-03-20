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
    
    train_images = torch.from_numpy(train_data["images"])
    train_poses = torch.from_numpy(train_data["poses"])
    N_images, Height, Weight, channal = train_images.shape
    K = torch.tensor([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                    ])
    
    if gpu:
        train_images = train_images.cuda()
        train_poses = train_poses.cuda()
        K = K.cuda()

        coarse_model = coarse_model.cuda()
        find_model = find_model.cuda()

    for i in range(epochs):
        # 一张图片一张图片训练
        index_image = random.sample(range(N_images), 1)
        train_image = train_images[index_image]
        train_pose = train_poses[index_image]

        render.get_rays(H, W, K, train_pose)
        grid_W, grid_H = torch.meshgrid(torch.arange(W), torch.arange(H))
        grid = torch.stack((grid_H, grid_W), dim=-1).transpose(1, 0)
        grid = grid.reshape((-1, 2))
        select_indexs = random.sample(grid.shape[0], N_rays)
        select_coords = grid[select_indexs].long()
        
        



    


    
    


