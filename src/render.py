import torch


def get_rays(Height, Width, K, c2w):
    X, Y = torch.meshgrid(torch.arange(Width), torch.arange(Height))
    rays_d = torch.stack(((Y - K[0][2]) / K[0][0], 
                         -(X - K[1][2]) / K[1][1], 
                         -torch.ones_like(X)), dim=-1)
    rays_d = rays_d.reshape((-1, 3))[..., None]
    rays_d = torch.bmm(c2w[:3, :3], rays_d)
    print(rays_d.shape)
    rays_o = c2w[:3, 4].expand(rays_d.shape)
    return rays_d, rays_o
    

