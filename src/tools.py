import json
import os
import imageio 
import numpy as np
import torch
import math
import render

# 平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])
# 绕x旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1]])
# 绕y旋转
rot_theta = lambda th: torch.Tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1]])


def pose_spherical(theta, phi, radius):
    '''
    这段代码是原来nerf-pytorch的，np和torch混着用，两个数据类型是有差异的，傻逼真是
    '''
    theta = torch.Tensor([theta])
    phi = torch.Tensor([phi])
    radius = torch.Tensor([radius])

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * torch.pi) @ c2w
    c2w = rot_theta(theta / 180. * torch.pi) @ c2w
    c2w = torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def set_default_datatype(datatype="32Float"):
    if datatype == "32Float":
        # 内存不够没法设置都在cuda上
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_dtype(torch.float32)
    else:
        # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        torch.set_default_dtype(torch.float64)


def gpu_run_environment(gpu=True):

    if not gpu:
        return False
    elif gpu and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return True
    else:
        return False


def read_datasets(datadir, dataconfig):
    splits = ['train', 'validate', 'test']
    all_datasets = {}
    for s in splits:
        with open(os.path.join(datadir, 
                               'transforms_{}.json'.format(s)), 'r') as fp:
            all_datasets[s] = json.load(fp)
    return_datasets = {}
    for s in splits:
        datasets = all_datasets[s]
        images = []
        poses = []
        for frame in datasets["frames"]:
            fname = os.path.join(datadir, frame['file_path'] + '.png')
            images.append(torch.Tensor(imageio.imread(fname)) / 255.)
            poses.append(torch.Tensor(frame['transform_matrix']))
        images = torch.stack(images, dim=0)
        poses = torch.stack(poses, dim=0)
        return_datasets[s] = {"images": images, "poses": poses}
    camera_angle_X = all_datasets["train"]["camera_angle_x"]
    H, W = images.shape[1:-1]
    focal = (W * 0.5) / np.tan(camera_angle_X * 0.5)
    K = torch.Tensor([
                     [focal, 0, 0.5 * W],
                     [0, focal, 0.5 * H],
                     [0, 0, 1]
                     ])
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) 
                                for angle in torch.linspace(-180, 180, 40 + 1)[: -1]], 0)
    min_bound, max_bound = get_bounding_box(datasets["frames"], H, W, K, 2.0, 6.0)
    return return_datasets, (H, W, focal), render_poses, min_bound, max_bound


def find_min_max_distance(pt, min_bound, max_bound):
    for i in range(3):
        if min_bound[i] > pt[i]:
            min_bound[i] = pt[i]
        if max_bound[i] < pt[i]:
            max_bound[i] = pt[i]
    return max_bound, min_bound


def get_bounding_box(frames, H, W, focal, near, far):
    min_bound = [math.inf, math.inf, math.inf]
    max_bound = [-math.inf, -math.inf, -math.inf]
    for frame in frames:
        pose = torch.Tensor(frame['transform_matrix'])
        rays_d, rays_o = render.get_rays(H, W, focal, pose)
        near_coords = rays_o + near * rays_d
        far_coords = rays_o + far * rays_d
        near_coords = near_coords.reshape(-1, 3)
        far_coords = far_coords.reshape(-1, 3)
        for i in [0, W - 1, H * W - W, H * W - 1]:
            min_point = near_coords[i]
            max_point = far_coords[i]
            max_bound, min_bound = find_min_max_distance(min_point, min_bound, max_bound)
            max_bound, min_bound = find_min_max_distance(max_point, min_bound, max_bound)
    return (torch.tensor(min_bound) - torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor(max_bound) + torch.tensor([1.0, 1.0, 1.0]))


def mse2psnr(X, gpu):
    device = 'cuda' if gpu else 'cpu'
    return -10. * torch.log(X) / torch.log(torch.Tensor([10.])).to(device)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)