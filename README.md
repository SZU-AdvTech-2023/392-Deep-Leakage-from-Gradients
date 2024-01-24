# Deep leakage from gradients combine FedADMM

本项目包含使用PyTorch实现的深度学习模型，结合DLG以及FedADMM代码（<https://github.com/YonghaiGong/FedADMM>）进行修改拓展，旨在研究并改善模型训练过程中的隐私保护机制。项目实现了DLG与ADMM在FL领域的结合以此产生DLG-ADMM，并对数据进行处理以进行隐私保护的实验。

## 环境依赖

*   Python 3.x
*   PyTorch 2.x
*   torchvision
*   NumPy
*   PIL
*   matplotlib
*   seaborn

## 使用方法

确保所有依赖都已正确安装，然后运行`DLGADMM_main`函数。

## 模型架构

项目沿用原论文中的LeNet网络，包含多个卷积层和全连接层，用于图像分类任务。

## 数据集

支持多种数据集，包括MNIST、CIFAR100和LFW。数据集将在第一次运行时自动下载。

## 结果输出

实验结果将保存在`results`目录下，包括损失值、均方误差（MSE）和图像重建的过程。

## 引用要求

如果您在研究中使用了本代码，请引用以下论文：

```markdown
@article{qiu2022decentralized,
  title={Decentralized Federated Learning for Industrial IoT With Deep Echo State Networks},
  author={Qiu, Wenqi and Ai, Wu and Chen, Huazhou and Feng, Quanxi and Tang, Guoqiang},
  journal={IEEE Transactions on Industrial Informatics},
  volume={19},
  number={4},
  pages={5849--5857},
  year={2022},
  publisher={IEEE}
}
```

## 许可证

该项目代码仅供学术研究使用，不得用于商业目的。在使用过程中，请遵守相应的许可证规定。

## 联系方式

若您在使用本代码过程中有任何疑问，请通过以下方式联系我：

电子邮件：[qiuwenqi788@gmail.com]()
