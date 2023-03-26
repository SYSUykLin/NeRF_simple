import tools
import train
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    
    dataset_name = ['lego', 'chair', 'ship']
    # 设置默认数据格式
    tools.set_default_datatype("32Float")
    is_gpu = tools.gpu_run_environment(True)
    datadir = ".\\dataset\\nerf_synthetic\\" + str(dataset_name[2])
    data_config = 'dataset\\' + str(dataset_name[2]) + '.txt'
    
    train.train_nerf(datadir=datadir,
                     dataconfig=data_config, 
                     dataname=str(dataset_name[2]),
                     gpu=is_gpu)
    