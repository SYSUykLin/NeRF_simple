# NeRF_simple
* 简单实现：paper，ECCV 2022 NeRF
* 设备：1060台式，6代i7，6G显存
* 数据：只选取了lego数据，跑了800x800整张图，和nerf-pytorch跑一半有区别。

为了减少占用现存做了几步操作：
* 没有像nerf-pytorch一样一开始把rays都放进cuda里面train，用到什么就to(cuda)什么，这样能够保证1024条射线能够训练起来
* 网络里面的relu改成inplace=True操作，不创作新变量
* 测试渲染全图的时候，分批渲染（nerf-pytorch也有），增加with no grad，并且删除无用变量和数据
* 测试渲染全图的时候渲染一部分就直接调到cpu里面不占内存
训练一张图片所用射线1024，测试一张图片所用射线6144。

* command：python run_nerf.py
* env：和nerf-pytorch一样的环境，但是torch用的1.13.0版本。如果是1.9版本会存在网络append
* 方法的问题，meshgrid的参数indexing的问题，自己调一下就行。
* Note：模型单纯的用800x800图片训练会陷入局部最优的问题，训练五次可能只有一两次是收敛的。
加上他的步长学习

效果：

https://user-images.githubusercontent.com/34080744/227677972-165d7c9d-6fb8-490c-99b8-e095a7e678cf.mp4

还是有点模糊，但是设备限制没法跑太多轮次了。
![nerf_simple_loss_pnsr](https://user-images.githubusercontent.com/34080744/227678042-040d5c10-3758-4a09-a964-a3043e0d531c.png)

***
增加了hash编码，instanNGP。
* 图片改成400x400
* 用xavier初始化网络
* 网络变小
* mesh的提取和上色
