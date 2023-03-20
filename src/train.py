import tools
import models


def train_nerf(datadir, dataconfig, gpu=True):
    print(f"use GPU : {gpu}")
    datasets, H, W, focal, render_poses = tools.read_datasets(datadir, dataconfig)
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
    train_data = datasets['train']
    val_data = datasets['validate']
    test_data = datasets['test']

    coordinate_embeddings = models.fourier_embedding(L=coordinate_L)
    direction_embeddings = models.fourier_embedding(L=direction_L)
    coord_input_dim = coordinate_embeddings.count_dim()
    direct_input_dim = direction_embeddings.count_dim()

    coarse_model = models.nerf(coord_input_dim, direct_input_dim, coordinate_embeddings, direction_embeddings)
    find_model = models.nerf(coord_input_dim, direct_input_dim, coordinate_embeddings, direction_embeddings)

    


    
    


