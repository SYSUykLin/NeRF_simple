import torch


def get_rays(Height, Width, K, c2w, gpu=True):
    # c2w[1, 3, 4]
    device = "cuda" if gpu else "cpu"
    X, Y = torch.meshgrid(torch.arange(Width), torch.arange(Height))
    X = X.t().to(device)
    Y = Y.t().to(device)
    # rays_d：[H, W, 3]
    rays_d = torch.stack([(X - K[0][2]) / K[0][0], 
                         -(Y - K[1][2]) / K[1][1], 
                         -torch.ones_like(X)], 
                         dim=-1)
    rays_d = torch.bmm(c2w[:, :3, :3].expand(Height * Width, 3, 3), 
                       rays_d.reshape((-1, 3))[..., None]).squeeze(-1)
    rays_d = rays_d.reshape(Height, Width, -1)
    rays_o = c2w[0, :3, -1].expand(rays_d.shape)
    return rays_d, rays_o


def coarse_samples(N_rays, near, far, N_samples, 
                   perturb=True, distribution="avg"):
    assert distribution in ['avg', 'gaussion']

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    z_vals = near + t_vals * (far - near)
    z_vals = z_vals.expand(N_rays, N_samples)
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        lower = torch.cat([z_vals[..., :1], mids], -1)
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        
        if distribution == 'avg':
            t_randoms = torch.rand(z_vals.shape)
        elif distribution == 'gaussion':
            t_randoms = torch.randn(z_vals.shape)
        z_vals = lower + t_randoms * (upper - lower)
    return z_vals


def find_samples():
    pass


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

def render_rays(z_vals, rays_d, rays_o):
    '''
    z_vals: [N_rays, N_samples]
    rays_d: [N_rays, 3]
    rays_o: [N_rays, 3]
    '''
    pass

    
    




    