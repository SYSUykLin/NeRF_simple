import tools


def train_nerf(datadir, dataconfig, gpu=True):
    print(f"use GPU : {gpu}")
    datasets, poses, H, W, focal, render_poses = tools.read_datasets(datadir, dataconfig)
    print(f"images shape : {H, W}")

    '''
    1.创建NeRF
    2.渲染
    3.训练
    '''
    


