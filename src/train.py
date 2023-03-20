import tools
import models
import torch
from torch.utils.data import RandomSampler


def train_nerf(datadir, dataconfig, gpu=True):
    print(f"use GPU : {gpu}")
    datasets, (H, W, focal), render_poses = tools.read_datasets(datadir, dataconfig)
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

    coarse_model = models.nerf(coord_input_dim, direct_input_dim, coordinate_embeddings, direction_embeddings)
    find_model = models.nerf(coord_input_dim, direct_input_dim, coordinate_embeddings, direction_embeddings)
    
    train_images = train_data["images"]
    train_poses = train_data["poses"]
    print(train_images[1][400, 400])
    N_images, Height, Weight, channal = train_images.shape
    for i in range(epochs):
        # 一张图片一张图片训练
        train_images = train_images[torch.randperm(N_images)]
        print(train_images[1][400, 400])
        import sys
        sys.exit()
        train_poses = train_poses[torch.randperm(N_images)]
        


    


    
    


