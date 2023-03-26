import torch.nn as nn
import torch


class nerf_ngp(nn.Module):
    def __init__(self,
                 coordinate_input_dim,
                 direction_input_dim,
                 first_depth=1,
                 second_depth=3,
                 hidden_size=64,
                 output_hidden=15):
        super(nerf_ngp, self).__init__()
        self.coordinate_input_dim = coordinate_input_dim
        self.direction_input_dim = direction_input_dim
        self.first_depth = first_depth
        self.second_depth = second_depth
        self.hidden_size = hidden_size
        self.output_hidden = output_hidden
        
        first_models = [nn.Linear(self.coordinate_input_dim, self.hidden_size), nn.ReLU(inplace=True)]
        for i in range(self.first_depth - 1):
            first_models.append(nn.Linear(self.hidden_size, self.hidden_size)) 
            # 这个得加，变成替换操作，要不然内存太大了
            first_models.append(nn.ReLU(inplace=True))
        first_models.append(nn.Linear(self.hidden_size, 1 + self.output_hidden))
        self.first_model = nn.Sequential(*first_models)
        
        second_models = [nn.Linear(self.output_hidden + self.direction_input_dim, self.hidden_size), nn.ReLU(inplace=True)]
        for i in range(self.second_depth - 1):
            second_models.append(nn.Linear(self.hidden_size, 
                                           self.hidden_size))
            second_models.append(nn.ReLU(inplace=True))
        second_models.append(nn.Linear(self.hidden_size, 3))
        self.second_model = nn.Sequential(*second_models)

        # 初始化
        for name, param in self.first_model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param)
                # nn.init.xavier_normal_(param, gain=1.0)
        for name, param in self.second_model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param)
                nn.init.xavier_normal_(param, gain=1.0)
        
    def forward(self, X_coordinate, Y_coordinate, keepmap):
        # X_coordinate, keepmap = self.embeddings_coordi(X_coordinate)
        N_rays, N_samples, dim = X_coordinate.shape
        first_hidden = self.first_model(X_coordinate)
        sigma = first_hidden[..., 0]
        first_hidden = first_hidden[..., 1:]
        second_hidden = torch.cat([first_hidden, Y_coordinate], dim=-1)
        color = self.second_model(second_hidden)
        raw = torch.cat([color, sigma[..., None]], dim=-1)
        raw = raw.reshape(-1, 4)
        # 盒子外面的密度设置为0
        raw[~keepmap, -1] = 0
        raw = raw.reshape(N_rays, N_samples, -1)
        return raw


class nerf(nn.Module):
    def __init__(self,
                 coordinate_input_dim,
                 direction_input_dim,
                 embeddings_coordi,
                 embeddings_direct,
                 first_depth=4,
                 second_depth=4,
                 hidden_size=64,
                 output_hidden=128):
        super(nerf, self).__init__()
        self.coordinate_input_dim = coordinate_input_dim
        self.direction_input_dim = direction_input_dim
        self.first_depth = first_depth
        self.second_depth = second_depth
        self.hidden_size = hidden_size
        self.output_hidden = output_hidden
        self.embeddings_coordi = embeddings_coordi
        self.embeddings_direct = embeddings_direct

        self.first_model = nn.Sequential(nn.Linear(self.coordinate_input_dim, 
                                                   self.hidden_size), 
                                         nn.ReLU(inplace=True))
        for i in range(self.first_depth - 1):
            self.first_model.append(nn.Linear(self.hidden_size, 
                                              self.hidden_size)) 
            # 这个得加，变成替换操作，要不然内存太大了
            self.first_model.append(nn.ReLU(inplace=True))
        
        self.second_model = nn.Sequential(nn.Linear(self.hidden_size + 
                                                    self.coordinate_input_dim, 
                                                    self.hidden_size), 
                                          nn.ReLU(inplace=True))
        for i in range(self.second_depth - 1):
            self.second_model.append(nn.Linear(self.hidden_size, 
                                               self.hidden_size))
            self.second_model.append(nn.ReLU(inplace=True))
        
        self.sigma_model = nn.Linear(self.hidden_size, 1)
        self.feature_model = nn.Linear(self.hidden_size, self.hidden_size)
        self.feature_direct_model = nn.Linear(self.hidden_size + 
                                              self.direction_input_dim, 
                                              self.hidden_size // 2)
        self.color_model = nn.Linear(self.hidden_size // 2, 3)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X_coordinate, Y_coordinate):
        X_coordinate = self.embeddings_coordi(X_coordinate)
        Y_coordinate = self.embeddings_direct(Y_coordinate)
        first_hidden = self.first_model(X_coordinate)
        first_hidden = torch.cat([first_hidden, X_coordinate], dim=-1)
        second_hidden = self.second_model(first_hidden)
        sigma = self.sigma_model(second_hidden)
        second_hidden = self.feature_model(second_hidden)
        second_hidden = torch.cat([second_hidden, Y_coordinate], dim=-1)
        third_hidden = self.feature_direct_model(second_hidden)
        third_hidden = self.relu(third_hidden)
        color = self.color_model(third_hidden)
        raw = torch.cat([color, sigma], dim=-1)
        return raw


class fourier_embedding(nn.Module):
    def __init__(self, L, include_X=True):
        super(fourier_embedding, self).__init__()
        self.L = L
        self.include = include_X

    def forward(self, X):
        # X:[batch_size, N_samples, 3]
        X_include = X.contiguous()
        X_return = []

        L_list = torch.arange(0, self.L, 1)
        # [2^0,2^1, ..., 2^L-1]
        embeddings = torch.pow(2, L_list) * torch.pi
        for embe in embeddings:
            # 2^l
            X = X_include * embe
            # [batch_size, N_samples, 6]
            X = self.embe_func(X)
            X_return.append(X)
        if self.include:
            X_return.append(X_include)
        X_return = torch.cat(X_return, dim=-1)
        return X_return

    def embe_func(self, X):
        return torch.cat([torch.sin(X), torch.cos(X)], dim=-1)
    
    def count_dim(self):
        cood_dim = self.L * 2 * 3
        if self.include:
            cood_dim += 3
        return cood_dim
    

class hash_embedding(nn.Module):
    def __init__(self, bounding_box, T=19, L=16, features_size=2, 
                 min_resolution=16, finest_resolution=512):
        super(hash_embedding, self).__init__()
        self.T = 2**T
        self.log2T = T
        self.levels = L
        self.features_size = features_size
        self.min_resolution = torch.Tensor([min_resolution])
        self.max_resolution = torch.Tensor([finest_resolution])
        self.bounding_box = bounding_box
        self.offset = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                                   device='cuda')
        self.b = torch.exp((torch.log(self.max_resolution) -
                            torch.log(self.min_resolution)) /
                           (self.levels - 1))
        self.pi = torch.tensor([1, 2654435761, 805459861])
        
        # 这个写法能兼容1.9的torch
        embeddings_layers = [nn.Embedding(self.T, self.features_size) for i in range(self.levels)]
        self.embeddings = nn.Sequential(*embeddings_layers)
        for i in range(self.levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def forward(self, x):
        # 归一化x坐标
        # x:[N_rays, N_samples, 3] -> x:[N_rays x N_samples, 3]
        # 返回的result_embeddings:[N_rays, N_samples, hidden_size]
        x_origin = x.contiguous()

        min_bound, max_bound = self.bounding_box[0], self.bounding_box[1]
        # min_bound:[N_rays x N_samples, 3]
        min_bound = min_bound[None, :].expand(x.shape).to(x.device)
        # max_bounc:[N_rays x N_samples, 3]
        max_bound = max_bound[None, :].expand(x.shape).to(x.device)
        keep_mask = x==torch.max(torch.min(x, max_bound), min_bound)
        if not torch.all(x <= max_bound) or not torch.all(x >= min_bound):
            # print("ALERT: some points are outside bounding box. Clipping them!")
            x = torch.clamp(x, min=min_bound, max=max_bound)
        # x:[N_rays x N_samples, 3] 归一化
        x = (x - min_bound) / (max_bound - min_bound)
        results_embeddings = []
        for level in range(self.levels):
            # resolution
            Nl = torch.floor(self.min_resolution * (self.b ** level)).to(x.device)
            # box_size:[N_rays x N_samples, 3]
            box_size = (max_bound - min_bound) / Nl
            # min_box:[N_rays x N_samples, 3]
            min_box_normalize_coords = torch.floor(x * Nl).int()
            min_box_coords = min_box_normalize_coords * box_size + min_bound
            max_box_coords = min_box_coords + torch.Tensor([1.0, 1.0, 1.0]).to(x.device) * box_size
            # offset:[1, 8, 3] min_box[:, None, :]:[N_rays x N_samples, 1, 3]
            box_indexs = min_box_normalize_coords[:, None, :] + self.offset

            hash_embedding_indexs = self.hash_code(box_indexs)
            voxel_embedds = self.embeddings[level](hash_embedding_indexs)
            # 注意这里传进去的x_origin要求是原始的坐标，不是归一化之后的坐标
            context_embedding = self.trilinear_interp(x_origin, min_box_coords, max_box_coords, voxel_embedds)
            results_embeddings.append(context_embedding)
        results_embeddings = torch.cat(results_embeddings, dim=-1)

        # 这个keep_mask不是很重要
        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return results_embeddings, keep_mask
    
    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def hash_code(self, box_indexs):
        '''
        box_indexs:[N_rays x N_samples, 8, 3]
        '''
        xor_result = torch.zeros_like(box_indexs)[..., 0]
        for i in range(box_indexs.shape[-1]):
            xor_result ^= box_indexs[..., i] * self.pi[i]
        return torch.tensor((1 << self.log2T) - 1).to(xor_result.device) & xor_result

    def count_dim(self):
        return self.features_size * self.levels


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def count_dim(self):
        return self.out_dim

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result







