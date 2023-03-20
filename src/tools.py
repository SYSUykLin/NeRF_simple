import json
import os
import imageio 
import numpy as np
import torch

# 平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()
# 绕x旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()
# 绕y旋转
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def set_default_datatype(datatype="32Float"):
    if datatype == "32Float":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')


def gpu_run_environment(gpu=True):
    if not gpu:
        return False
    elif gpu and torch.cuda.is_available():
        return True


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
            images.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        images = (np.array(images) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        return_datasets[s] = {"images": images, "poses": poses}
    camera_angle_X = all_datasets["train"]["camera_angle_x"]
    H, W = images.shape[1:-1]
    focal = (W * 0.5) / np.tan(camera_angle_X * 0.5)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    return return_datasets, (H, W, focal), render_poses
