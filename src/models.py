import torch.nn as nn
import torch


class nerf(nn.Module):
    def __init__(self, 
                 coordinate_input_dim,
                 direction_input_dim,
                 embeddings_coordi,
                 embeddings_direct,
                 first_depth=4, 
                 second_depth=4, 
                 hidden_size=256, 
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
    def __init__(self, bounding_box, T=19, L=16, features_size=2, min_resolution=16, finest_resolution=512):
        super(hash_embedding, self).__init__()
        self.T = 2**T
        self.levels = L
        self.features_size = features_size
        self.min_resolution = torch.Tensor([min_resolution])
        self.max_resolution = torch.Tensor([finest_resolution])
        self.bounding_box = bounding_box
        self.offset = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                                   device='cuda')
        self.b = torch.exp((torch.log(self.max_resolution) - torch.log(self.min_resolution)) / (self.levels - 1))
        self.pi = torch.tensor([1, 2654435761, 805459861])
        
        # 这个写法能兼容1.9的torch
        embeddings_layers = [nn.Embedding(self.T, self.features_size) for i in range(self.levels)]
        self.embeddings = nn.Sequential(*embeddings_layers)

    def forward(self, x):
        # 归一化x坐标
        min_bound, max_bound = self.bounding_box[0], self.bounding_box[1]
        min_bound = min_bound[None, :].expand(x.shape)
        max_bound = max_bound[None, :].expand(x.shape)
        x = (x - min_bound) / (max_bound - min_bound)
        for level in range(self.levels):
            Nl = torch.floor(self.min_resolution * (self.b ** level))
            box_size = (max_bound - min_bound) / Nl
            min_box = torch.floor(x * Nl).int()
            box_indexs = min_box[None, ...] + self.offset
            self.hash_code(box_indexs)

    def hash_code(self, box_indexs):
        print(box_indexs.shape)
        pass   

    def count_dim(self):
        return self.features_size * self.levels






