基于mnist数据集的distilling过程实现

1.mnist_conv2D为搭建的大模型；

2.frozen保存其网络模型结构；

3.mnist_distilling为小模型，接受大模型的softtarget进行训练．
