import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import imageio
import tools
import numpy as np
from PIL import Image
import cv2


def get_rays(Height, Width, K, c2w, gpu=True):
    # c2w[1, 3, 4]
    c2w = c2w.reshape(-1, c2w.shape[-1])
    device = "cuda" if gpu else "cpu"
    X, Y = torch.meshgrid(torch.arange(Width), torch.arange(Height))
    X = X.t().to(device)
    Y = Y.t().to(device)
    # rays_d：[H, W, 3]
    rays_d = torch.stack([(X - K[0][2]) / K[0][0], 
                         -(Y - K[1][2]) / K[1][1], 
                         -torch.ones_like(X)], 
                         dim=-1)
    rays_d = torch.bmm(c2w[:3, :3].expand(Height * Width, 3, 3), 
                       rays_d.reshape((-1, 3))[..., None]).squeeze(-1)
    rays_d = rays_d.reshape(Height, Width, -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_d, rays_o


def coarse_samples(N_rays, near, far, N_samples, 
                   perturb=True, distribution="avg", gpu=True):
    assert distribution in ['avg', 'gaussion']
    
    device = 'cuda' if gpu else 'cpu'
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(device)
    z_vals = near + t_vals * (far - near)
    z_vals = z_vals.expand(N_rays, N_samples)
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        lower = torch.cat([z_vals[..., :1], mids], -1)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        
        if distribution == 'avg':
            t_randoms = torch.rand(z_vals.shape).to(device)
        elif distribution == 'gaussion':
            t_randoms = torch.randn(z_vals.shape).to(device)
        z_vals = lower + t_randoms * (upper - lower)
    return z_vals.to(device)


def fine_samples(weights, rays_d, z_vals, N_importance, perturb=True, gpu=True):
    '''
    z_vals: [N_rays, N_samples]
    weights: [N_rays, N_samples]
    '''
    device = 'cuda' if gpu else "cpu"
    # 不加会出现nan的情况
    weights = weights + 1e-5
    # 计算中点 [N_rays, N_samples-1]
    z_vals = z_vals.to(device)
    z_vals_mid = 0.5 * (z_vals[..., 1:] - z_vals[..., :-1])
    weights = weights[..., 1:-1]
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # [N_rays, N_samples-1]
    cdf = torch.cat([torch.zeros(cdf.shape[0], 1, device=device), cdf], dim=-1)
    
    # 均匀分出N_importance个点 [N_importance]
    u = torch.linspace(0.0, 1.0, N_importance, device=device)
    u = u[None, ...].expand(z_vals.shape[0], N_importance).contiguous()
    
    indexs = torch.searchsorted(cdf, u)
    lower = torch.max(torch.zeros(indexs.shape).to(device), indexs - 1)
    upper = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indexs).to(device), indexs)
    intervals = torch.stack([lower, upper], dim=-1).clone().detach()
    intervals = torch.as_tensor(intervals, dtype=torch.int64)
    # [N_rays, N_importance, N_samples-1]
    search_cdf = cdf[..., None, :].expand(cdf.shape[0], N_importance, cdf.shape[-1])
    search_bins = z_vals_mid[..., None, :].expand(z_vals_mid.shape[0], N_importance, z_vals_mid.shape[-1])
    # [N_rays, N_importance, 2]
    cdf_p = torch.gather(search_cdf, dim=-1, index=intervals)
    bins_intervals = torch.gather(search_bins, dim=-1, index=intervals)
    intervals_weights = cdf_p[..., 1] - cdf_p[..., 0]
    intervals_weights = torch.where(intervals_weights < 1e-5, torch.ones_like(intervals_weights), intervals_weights)
    t_vals = (u - cdf_p[..., 0]) / intervals_weights
    z_vals = bins_intervals[..., 0] + t_vals * (bins_intervals[..., 1] - bins_intervals[..., 0])
    return z_vals
    

def generate_raw(z_vals, model, rays_d, rays_o, gpu=True):
    '''
    z_vals: [N_rays, N_samples]
    rays_d: [N_rays, 3]
    rays_o: [N_rays, 3]
    '''
    device = 'cuda' if gpu else "cpu"
    z_vals = z_vals.to(device)
    coordinates = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., 
                                                                       None]
    # viewdirs: [N_rays, 3]
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, :].expand(coordinates.shape)
    # [N_rays, N_samples, 4]
    raw_outputs = model(coordinates, viewdirs)

    return raw_outputs, viewdirs, coordinates


def render_rays(z_vals, rays_d, color, sigma, gpu=True):
    '''
    z_vals: [N_rays, N_samples]
    rays_d: [N_rays, 3]
    rays_o: [N_rays, 3]
    color: [N_rays, N_samples, 3]
    sigma: [N_rays, N_samples]
    '''
    inf = 1e10
    device = 'cuda' if gpu else "cpu"
    z_vals = z_vals.to(device)
    # z轴上的距离 [N_rays, N_samples-1]
    z_vals_distance = z_vals[..., 1:] - z_vals[..., :-1]
    # 转换成射线上的距离 [N_rays, N_samples-1]
    rays_distance = z_vals_distance * torch.norm(rays_d, dim=-1, keepdim=True) 
    # rays_disance [N_rays, N_samples]最后一列没用
    rays_distance = torch.cat([rays_distance, torch.Tensor([inf]).expand(rays_distance.shape[0], 1).to(device)], 
                              dim=-1)
    # alpha = 1 - exp(-sigma*delta) [N_rays, N_samples]
    alpha = 1 - torch.exp(-F.relu(sigma) * rays_distance)
    # 计算Ti = exp(-sum(sigma*delta))
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1) 
    T = torch.cat([torch.ones(rays_distance.shape[0], 1, device=device), T], dim=-1)
    T = T[:, :-1]
    weights = alpha * T
    rgb = torch.sum((weights[..., None] * color), dim=1)
    depth = torch.sum((weights[..., None] * z_vals[..., None]), dim=1)
    acc_map = torch.sum(weights, -1)
    rgb = rgb + (1. - acc_map[..., None])

    return rgb, depth, weights

def render_images(render_poses, H, W, K, near, far, N_rays, N_samples, 
                  N_importance, coarse_model, fine_model, gpu=True):
    device = "cuda" if gpu else "cpu"
    render_poses = render_poses.to(device)
    coarse_model.eval()
    fine_model.eval()
    renders_images = []
    render_depths = []
    for i, c2w in enumerate(tqdm(render_poses)):
        rays_d, rays_o = get_rays(H, W, K, c2w, gpu=True)
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        sh = H * W

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        images_render_rays = []
        depth_render_rays = []
        for i in range(0, sh, N_rays):
            select_rays_d = rays_d[i: i + N_rays]
            select_rays_o = rays_o[i: i + N_rays]
            near, far = near * torch.ones_like(select_rays_d[..., :1]).to(device), far * torch.ones_like(select_rays_o[..., :1]).to(device)
            
            z_vals = coarse_samples(N_rays, near, far, N_samples, gpu=gpu)
            raw, viewdirs, coordinates = generate_raw(z_vals, 
                                                      coarse_model, 
                                                      select_rays_d, 
                                                      select_rays_o)
            color = raw[..., :3]
            sigma = raw[..., -1]
            rgb_images, depth_images, weights = render_rays(z_vals, select_rays_d, color, sigma, gpu)
            z_vals_fine_sample = fine_samples(weights, select_rays_d, z_vals, N_importance, gpu)
            z_vals_fine_sample = z_vals_fine_sample.detach()
            z_vals_all, _ = torch.sort(torch.cat([z_vals, z_vals_fine_sample], -1), -1)
            raw_fine, viewdirs_fine, coordinates_fine = generate_raw(z_vals_all, 
                                                                     fine_model, 
                                                                     select_rays_d, 
                                                                     select_rays_o)
            color_fine = raw_fine[..., :3]
            sigma_fine = raw_fine[..., -1]
            rgb_images_fine, depth_images_fine, weights_fine = render_rays(z_vals_all, select_rays_d, 
                                                                           color_fine, sigma_fine, gpu)
            images_render_rays.append(rgb_images_fine)
            depth_render_rays.append(depth_images_fine)
        images = torch.cat(images_render_rays, dim=0).reshape(H, W, -1)
        depth = torch.cat(depth_render_rays, dim=0).reshape(H, W)
        renders_images.append(images.cpu().numpy())
        render_depths.append(depth.cpu().numpy())
    renders_images = torch.stack(renders_images, dim=0)
    imageio.mimwrite(os.path.join('dataset\\nerf_synthetic\\lego\\logs', 'fine_network_video.mp4'), 
                     tools.to8b(renders_images), fps=30, quality=8)
    depth_map_visualization(render_depths)
    
    return renders_images, render_depths

def depth_map_visualization(depth_images):
    for image in depth_images:
        im_color = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=15), cv2.COLORMAP_JET)
        im = Image.fromarray(im_color)
        print(im.shape)
        