import tools
import train
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':

    # 设置默认数据格式
    tools.set_default_datatype("32Float")
    is_gpu = tools.gpu_run_environment()
    datadir = ".\\dataset\\nerf_synthetic\\lego"
    data_config = 'dataset\\lego.txt'
    
    train.train_nerf(datadir=datadir,
                     dataconfig=data_config, 
                     gpu=is_gpu)
    