# MLR-SNet: Transferable LR Schedules for Heterogeneous Tasks

This is an official PyTorch implementation of [MLR-SNet: Transferable LR Schedules for Heterogeneous Tasks](https://arxiv.org/abs/2007.14546). Please contact: Jun Shu (xjtushujun@gmail.com); Deyu Meng(dymeng@mail.xjtu.edu.cn).

```
@journal{shu2021mlrsnet,
    title     = {MLR-SNet: Transferable LR Schedules for Heterogeneous Tasks},
    author    = {Shu, Jun and Zhu, Yanwen and Zhao, Qian and Meng, Deyu and Xu, Zongben},
    year      = {2021}
} 
```

## Overview

<img align="middle" width="100%" src="mlrsnet.png"> 

The learning rate (LR) is one of the most important hyperparameters in stochastic gradient descent (SGD) algorithm for training deep neural networks (DNN). However, current hand-designed LR schedules need to manually pre-specify a fixed form, which limits their ability to adapt to practical non-convex optimization problems due to the significant diversification of training dynamics. Meanwhile, it always needs to search proper LR schedules from scratch for new tasks, which, however, are often largely different with task variations, like data modalities, network architectures, or training data capacities. To address this learning-rate-schedule setting issues, we propose to parameterize LR schedules with an explicit mapping formulation, called \textit{MLR-SNet}. The learnable parameterized structure brings more flexibility for MLR-SNet to learn a proper LR schedule to comply with the training dynamics of DNN. Image and text classification benchmark experiments substantiate the capability of our method for achieving proper LR schedules. Moreover, the explicit parameterized structure makes the meta-learned LR schedules capable of being transferable and plug-and-play, which can be easily generalized to new heterogeneous tasks. We transfer our meta-learned MLR-SNet to query tasks like different training epochs, network architectures, data modalities, dataset sizes from the training ones, and achieve comparable or even better performance compared with hand-designed LR schedules specifically designed for the query tasks. The robustness of MLR-SNet is also substantiated when the training data are biased with corrupted noise. We further prove the convergence of the SGD algorithm equipped with LR schedule produced by our MLR-Net, with the convergence rate comparable to the best-known ones of the algorithm for solving the problem.


## Prerequisites
- Python 3.7 (Anaconda)
- PyTorch >= 1.2.0
- Torchvision >= 0.2.1


## Meta-Train: adapting to the training dynamics of DNN (Section 4.1)
Please use meta-train sub-folder to meta-learn the MLR-SNet meta-model. Here is an example for image-classification:
```bash
python meta-train.py --network resnet --dataset cifar10 --lr 1e-3
```
The lr is the learning rate of Adam meta-optimizer, we suggest to set $1e-3$, which has also been used for the Adam optimizer in Pytorch. Details can refer to Section 4.1.3 and Fig.6 of the main paper.

<img align="middle" width="100%" src="image.png"> 
<img align="middle" width="100%" src="text.png"> 

## Meta-Test: generalization to new heterogeneous tasks (Section 4.2)
Please use meta-test sub-folder to evaluate the transferability and generalization capability of the LR Schedules Meta-learned by MLR-SNe. We also provide the MLR-SNet we learned in the meta-test sub-folder. [mlr_snet 1.pth](https://github.com/xjtushujun/MLR-SNet/blob/main/meta-test/mlr_snet%201.pth) , [mlr_snet 100.pth](https://github.com/xjtushujun/MLR-SNet/blob/main/meta-test/mlr_snet%20100.pth), [mlr_snet 200.pth](https://github.com/xjtushujun/MLR-SNet/blob/main/meta-test/mlr_snet%20200.pth). Here is an example for transfer learned MLR-SNet to different network architectures setting.
```bash
python meta-test.py --network shufflenetv2 --dataset cifar10
```

<img align="middle" width="100%" src="datasets.png"> 
<img align="middle" width="100%" src="network.png"> 



