# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import time
import seaborn as sns
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
from update_delta import update
print(torch.__version__, torchvision.__version__)


class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def main():
    seed = 4123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    lr = 0.1
    dataset = 'cifar100'
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/ADMM_four_test_%s_%s_%s' % (dataset, seed, lr)).replace('\\', '/')

    num_dummy = 1
    Iteration = 300
    num_exp = 1

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        #dst = datasets.LFWPairs(data_path, download=True)
        lfw_path = os.path.join(root_path, '../data/lfw-py/lfw_funneled')
        dst = lfw_dataset(lfw_path, shape_img)
    else:
        exit('unknown dataset')

    ''' Train data '''
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        net.apply(weights_init)
        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))
        net.train()

        weights = copy.deepcopy(net.state_dict())
        alpha = {}
        for key in weights.keys():
            alpha[key] = torch.zeros_like(weights[key])

        for method in ['DLG_admm', 'iDLG_admm','iDLG', 'DLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)

            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient ——DLG
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            if method == "DLG_admm":
                # compute original ADMM gradient
                gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)
                original_delta_gd = update(gt_data, gt_onehot_label, False)
                dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)

            elif method == "iDLG_admm":
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)
                original_delta_gd = update(gt_data, gt_onehot_label, False)
                label_pred_ad = torch.argmin(torch.sum(original_delta_gd[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)
                label_pred_onehot_ad = label_to_onehot(label_pred_ad, num_classes=num_classes)

            elif method == "DLG":
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)

            elif method == "iDLG":
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            loss_plot = []
            mses = []
            mse_plot = []
            train_iters = []
            print('lr =', lr)

            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == "DLG_admm":
                        dummy_onehot_label = F.softmax(dummy_label, dim=-1).requires_grad_(True)
                        dummy_dy_dx = update(dummy_data, dummy_onehot_label, True)

                        dummy_dy_dx = [gx.clone().detach().requires_grad_(True) for gx in dummy_dy_dx]
                        grad_diff_accumulator = sum(
                            ((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, original_delta_gd))

                        grad_diff_accumulator.backward()

                        return grad_diff_accumulator

                    elif method == "iDLG_admm":
                        dummy_dy_dx = update(dummy_data, label_pred_onehot_ad, True)
                        dummy_dy_dx = [gx.clone().detach().requires_grad_(True) for gx in dummy_dy_dx]
                        grad_diff_accumulator = sum(
                            ((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, original_delta_gd))

                        grad_diff_accumulator.backward()

                        return grad_diff_accumulator

                    elif method == "DLG":
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff_accumulator = 0

                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff_accumulator += ((gx - gy) ** 2).sum()

                        grad_diff_accumulator.backward()

                        return grad_diff_accumulator

                    elif method == "iDLG":
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                        grad_diff = 0

                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                optimizer.step(closure)

                # 在每次迭代后添加以下代码以裁剪梯度
                #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 1.0是梯度的最大范数，可以根据需要调整

                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())

                # print(iters, "%.4f" % current_loss.item())
                # history_ADMM.append(tt(dummy_data[0].cpu()))

                if iters % int(Iteration / 30) == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)
                    loss_plot.append(current_loss)
                    mse_plot.append(torch.mean((dummy_data-gt_data)**2).item())

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')

                        if method == 'DLG':
                            image_path = '%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx])
                            plt.savefig(image_path)
                            plt.close()
                            #images.append(imageio.imread(image_path))

                        elif method == 'DLG_admm':
                            image_path = '%s/DLG_admm_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx])
                            plt.savefig(image_path)
                            plt.close()
                            #images.append(imageio.imread(image_path))

                        elif method == 'iDLG_admm':
                            image_path = '%s/iDLG_admm_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx])
                            plt.savefig(image_path)
                            plt.close()

                        elif method == "iDLG":
                            image_path = '%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx])
                            plt.savefig(image_path)
                            plt.close()

                    # gif_path = '%s/%s_%05d_%s.gif' % (save_path, imidx_list, imidx_list[imidx], method)
                    # imageio.mimsave(gif_path, images)
                    #print('Saved gif:', gif_path)

                    if current_loss < 0.0000001:  # converge
                        break

            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history_iters, loss_plot, label='Loss', linewidth=2, marker='o')
                ax.set_title('Loss vs Iterations', fontsize=16)
                ax.set_xlabel('Iterations', fontsize=14)
                ax.set_ylabel('Loss', fontsize=14)
                ax.legend()
                plt.savefig('%s/%s_%05d_%s.png' % (save_path, imidx_list, imidx_list[imidx], method))
                plt.close()

            with sns.axes_style("whitegrid"):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history_iters, mse_plot, label='MSE', linewidth=2, marker='o')
                ax.set_title('MSE vs Iterations', fontsize=16)
                ax.set_xlabel('Iterations', fontsize=14)
                ax.set_ylabel('MSE', fontsize=14)
                ax.legend()
                plt.savefig('%s/%s_%05d_%s_mse.png' % (save_path, imidx_list, imidx_list[imidx], method))
                plt.close()

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses

            elif method == 'DLG_admm':
                loss_DLG_admm = losses
                label_DLG_admm = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG_admm = mses

            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

            elif method == 'iDLG_admm':
                loss_iDLG_admm = losses
                label_iDLG_admm = label_pred_ad.item()
                mse_iDLG_admm = mses

        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_DLG_admm:', loss_DLG_admm[-1],'loss_iDLG:', loss_iDLG[-1], 'loss_iDLG_admm',loss_iDLG_admm[-1])
        print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_DLG_admm[-1], 'mse_iDLG:', mse_iDLG[-1],'mse_iDLG_admm:', mse_iDLG_admm[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_DLG_admm:', label_DLG_admm, 'lab_iDLG:', label_iDLG,'lab_iDLG_admm:', label_iDLG_admm)

        print('----------------------\n\n')


if __name__ == '__main__':
    main()










