import tools
import models
import torch
import random
import render
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())   


def train_nerf(datadir, dataconfig, gpu=True):
    print(f"use GPU : {gpu}")
    writer = SummaryWriter('dataset\\nerf_synthetic\\lego\\logs' + str(TIMESTAMP))
    datasets, (H, W, focal), render_poses = tools.read_datasets(datadir, 
                                                                dataconfig)
    print(f"images shape : {H, W}")

    '''
    1.创建NeRF
    2.渲染
    3.训练
    '''

    epochs = 300000
    test_iter = 10000
    coordinate_L = 10
    direction_L = 4
    N_rays = 1024
    rander_rays = 4096
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
    fine_model = models.nerf(coord_input_dim, direct_input_dim, 
                             coordinate_embeddings, direction_embeddings)
    grad_vars = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=5e-4, betas=(0.9, 0.999))
    
    # 所有的训练数据
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
        coarse_model = coarse_model.cuda()
        fine_model = fine_model.cuda()
    
    epochs_lists = [i for i in range(epochs // test_iter)]
    for global_epoch in range(len(epochs_lists)):
        for epoch in tqdm(range(test_iter), colour='GREEN', ncols=80):
            # 一张图片一张图片训练
            index_image = random.sample(range(N_images), 1)
            train_image = train_images[index_image]
            train_image = train_image[..., :3] * train_image[..., -1:] + (1. - train_image[..., -1:])
            _, Height, Width, channal = train_image.shape
            train_pose = train_poses[index_image]

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
            
            if gpu:
                targets_image = targets_image.cuda()
                train_pose = train_pose.cuda()
            z_vals = render.coarse_samples(N_rays, near, far, N_samples, gpu=gpu)
            raw, viewdirs, coordinates = render.generate_raw(z_vals, 
                                                             coarse_model, 
                                                             select_rays_d, 
                                                             select_rays_o)
            # viewdirs：归一化后的方向
            color = raw[..., :3]
            sigma = raw[..., -1]
            rgb_images, depth_images, weights = render.render_rays(z_vals, select_rays_d, color, sigma, gpu)
            z_vals_fine_sample = render.fine_samples(weights, select_rays_d, z_vals, N_importances, gpu=True)
            z_vals_fine_sample = z_vals_fine_sample.detach() 
            z_vals_all, _ = torch.sort(torch.cat([z_vals, z_vals_fine_sample], -1), -1)

            raw_fine, viewdirs_fine, coordinates_fine = render.generate_raw(z_vals_all, 
                                                                            fine_model, 
                                                                            select_rays_d, 
                                                                            select_rays_o)
            color_fine = raw_fine[..., :3]
            sigma_fine = raw_fine[..., -1]
            rgb_images_fine, depth_images_fine, weights_fine = render.render_rays(z_vals_all, select_rays_d, 
                                                                                  color_fine, sigma_fine, gpu)
            
            optimizer.zero_grad()
            loss_mse = torch.nn.MSELoss(reduction='mean')
            loss_images = loss_mse(rgb_images_fine, targets_image)
            psnr = tools.mse2psnr(loss_images, gpu)
            loss_images.backward()
            torch.nn.utils.clip_grad_norm_(grad_vars, 10) 
            optimizer.step()

            # tqdm.write(f"loss : {loss_images}, psnr : {psnr.item()}")
            writer.add_scalar("Loss", loss_images, 
                              global_step=global_epoch * test_iter + epoch)
            writer.add_scalar("PSNR", psnr.item(), 
                              global_step=global_epoch * test_iter + epoch)
        torch.cuda.empty_cache()
        render.render_images(render_poses, H, W, K, near, far, 
                             rander_rays, N_samples, 
                             N_importances, coarse_model, fine_model, global_epoch * test_iter, gpu)    
            
        
    writer.close()
        
        



    


    
    


