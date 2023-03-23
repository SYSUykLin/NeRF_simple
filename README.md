# NeRF_simple
简单实现：paper，ECCV 2022 NeRF
设备：1060台式，6代i7，6G显存
数据：只选取了lego数据，跑了800x800整张图，和nerf-pytorch跑一半有区别。

为了减少占用现存做了几步操作：
* 没有像nerf-pytorch一样一开始把rays都放进cuda里面train，用到什么就to(cuda)什么，这样能够保证1024条射线能够训练起来
* 网络里面的relu改成inplace=True操作，不创作新变量
* 测试渲染全图的时候，分批渲染（nerf-pytorch也有），增加with no grad，并且删除无用
* 测试渲染全图的时候渲染一部分就直接调到cpu里面不占内存
训练一张图片所用射线1024，测试一张图片所用射线6144。

command：python run_nerf.py
env：和nerf-pytorch一样的环境，但是torch用的1.13.0版本。如果是1.9版本会存在网络append
方法的问题，meshgrid的参数indexing的问题，自己调一下就行。

