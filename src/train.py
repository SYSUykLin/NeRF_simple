import tools
import models
import torch
import random
import render
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())   


def train_nerf(datadir, dataconfig, dataname, gpu=True):
    print(f"use GPU : {gpu}")
    writer = SummaryWriter('dataset\\nerf_synthetic\\' + dataname + '\\logs' + str(TIMESTAMP))
    datasets, (H, W, focal), render_poses, min_bound, max_bound = tools.read_datasets(datadir, 
                                                                                      dataconfig)
    bounding_boxes = (min_bound, max_bound)
    print(f"images shape : {H, W}")

    '''
    1.创建NeRF
    2.渲染
    3.训练
    '''

    epochs = 300000
    test_iter = 5000
    coordinate_L = 10
    direction_L = 4
    N_rays = 1024
    rander_rays = 4096
    N_samples = 64
    N_importances = 128
    near = 2.
    far = 6.
    train_data = datasets['train']
    # val_data = datasets['validate']
    # test_data = datasets['test']
    
    coordinate_embeddings = models.hash_embedding(bounding_boxes)
    # coordinate_embeddings = models.fourier_embedding(L=10)
    direction_embeddings = models.SHEncoder()
    coord_input_dim = coordinate_embeddings.count_dim()
    direct_input_dim = direction_embeddings.count_dim()
    # coarse_model = models.nerf(coord_input_dim, direct_input_dim, 
    #                            coordinate_embeddings, direction_embeddings)
    # fine_model = models.nerf(coord_input_dim, direct_input_dim, 
    #                          coordinate_embeddings, direction_embeddings)
    coarse_model = models.nerf_ngp(coord_input_dim, direct_input_dim)
    fine_model = models.nerf_ngp(coord_input_dim, direct_input_dim)
    grad_vars = list(coarse_model.parameters()) + list(fine_model.parameters())
    embedding_params = list(coordinate_embeddings.parameters())
    grad_vars = grad_vars + embedding_params
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
        coordinate_embeddings = coordinate_embeddings.cuda()
        direction_embeddings = direction_embeddings.cuda()
        coarse_model = coarse_model.cuda()
        fine_model = fine_model.cuda()
    
    lr_global_epochs = 0

    epochs_lists = [i for i in range(epochs // test_iter)]
    for global_epoch in range(len(epochs_lists)):
        for epoch in tqdm(range(test_iter), colour='GREEN', ncols=80):
            # 一张图片一张图片训练
            index_image = random.sample(range(N_images), 1)
            train_image = train_images[index_image]
            train_image = train_image[..., :3] * train_image[..., -1:] + (1. - train_image[..., -1:])
            _, Height, Width, channal = train_image.shape
            train_pose = train_poses[index_image, :3, :4]

            rays_d, rays_o = render.get_rays(H, W, K, train_pose, gpu)
            grid_W, grid_H = torch.meshgrid(torch.linspace(0, Width - 1, Width), torch.linspace(0, Height - 1, Height))
            grid = torch.stack((grid_H.t(), grid_W.t()), dim=-1).long()  
            if epoch < 500:
                dH = int(H // 2 * 0.5)
                dW = int(W // 2 * 0.5)
                grid = torch.stack(torch.meshgrid(torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH), 
                                                  torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                                                  ), -1).long()    
            grid = grid.reshape((-1, 2))

            select_indexs = np.random.choice(grid.shape[0], size=[N_rays], replace=False)

            select_coords = grid[select_indexs]
            
            train_image = train_image.reshape(Height, Width, channal)
            select_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1], :]      
            select_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1], :]
            targets_image = train_image[select_coords[:, 0], select_coords[:, 1], :]
            
            targets_image = torch.Tensor(targets_image)
            if gpu:
                targets_image = targets_image.cuda()
                train_pose = train_pose.cuda()
            z_vals = render.coarse_samples(N_rays, near, far, N_samples, gpu=gpu)
            raw, viewdirs, coordinates = render.generate_raw(z_vals, 
                                                             coarse_model, 
                                                             select_rays_d, 
                                                             select_rays_o,
                                                             coordinate_embeddings, 
                                                             direction_embeddings)
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
                                                                            select_rays_o,
                                                                            coordinate_embeddings, 
                                                                            direction_embeddings)
            color_fine = raw_fine[..., :3]
            sigma_fine = raw_fine[..., -1]
            rgb_images_fine, depth_images_fine, weights_fine = render.render_rays(z_vals_all, select_rays_d, 
                                                                                  color_fine, sigma_fine, gpu)
            
            optimizer.zero_grad()
            loss_mse = torch.nn.MSELoss(reduction='mean')
            loss_fine_images = loss_mse(rgb_images_fine, targets_image)
            loss_coarse_images = loss_mse(rgb_images, targets_image)

            loss = loss_fine_images + loss_coarse_images

            psnr = tools.mse2psnr(loss_fine_images, gpu)
            loss.backward()
            optimizer.step()

            # 原生代码里面的步长优化
            lrate_decay = 250
            lrate = 5e-4
            decay_rate = 0.1
            decay_steps = lrate_decay * 1000
            new_lrate = lrate * (decay_rate ** ((lr_global_epochs) / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if epoch % 50 == 0:
                tqdm.write(f"loss : {loss}, psnr : {psnr.item()}")
            writer.add_scalar("Loss", loss, 
                              global_step=lr_global_epochs)
            writer.add_scalar("PSNR", psnr.item(), 
                              global_step=lr_global_epochs)
            lr_global_epochs += 1
        torch.cuda.empty_cache()
        torch.save(coordinate_embeddings, 'models\\coordinate_embeddings.pt')
        torch.save(direction_embeddings, 'models\\direction_embeddings.pt')
        torch.save(coarse_model, 'models\\coarse_model.pt')
        torch.save(fine_model, 'models\\fine_model.pt')
       
        render.render_images(render_poses, H, W, K, near, far, 
                             rander_rays, N_samples, 
                             N_importances, coarse_model, fine_model, 
                             coordinate_embeddings, direction_embeddings, 
                             global_epoch * test_iter, dataname, gpu)    
            
        
    writer.close()
        
        



    


    
    


