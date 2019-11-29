'''
mnist classification
'''

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import math
from ptflops import get_model_complexity_info

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)

EPOCH = 50
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_FASHIONMNIST = False

# DOWNLOAD_FASHIONMNIST = True
if not(os.path.exists('./test/')) or not os.listdir('./test/'):
    DOWNLOAD_FASHIONMNIST = True
test_data = torchvision.datasets.MNIST(root='./test/', train=False,transform=transforms.Compose([#transforms.ToPILImage(),
                                  #transforms.RandomRotation(15),
                                  # RandomShift(5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]), ## 有问题。
    download=DOWNLOAD_FASHIONMNIST
)
# test data
test_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=False, num_workers=2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 10),
            )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.features.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def mnist_vis(b_x, b_y):
    b_x, b_y = np.squeeze(b_x.data.numpy()), np.squeeze(b_y.data.numpy())
    plt.figure()
    
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.title(b_y[i])
        plt.imshow(b_x[i, :, :])
        print(b_x)
        plt.axis('off')
    plt.axis('off')
    plt.show()

def ohkpm(loss, top_k):
    
        ohkpm_loss = 0.
        topk_val, topk_idx = torch.topk(loss, k=top_k, dim=0, sorted=False)
        tmp_loss = torch.gather(loss, 0, topk_idx)

        ohkpm_loss = torch.unsqueeze(tmp_loss, dim=0).mean()

        return ohkpm_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    reload = True
    best_acc = 0.
    # from resnet_mnist import resnet_mnist
    # cnn = resnet_mnist().cuda()
    cnn = CNN()
    epoch = 0
    loss_avg = AverageMeter()
    flops, params = get_model_complexity_info(cnn, (28,28), as_strings=True, print_per_layer_stat=True)
    print("flops:  " + flops)
    print("params: " + params)
    cnn.cuda()
    
    model = torch.load('mnist_best9976.pth')
    cnn.load_state_dict(model['state_dict'])
    epoch = model['epoch']
    best_acc = model['acc']

    total = 0
    cnn.eval()
    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        test_pred = cnn(b_x)
        test_pred = test_pred.cpu()
        test_pred = torch.max(test_pred, 1)[1].data.squeeze().numpy()

        b_x = b_x.cpu()
        b_y = b_y.cpu()
        n = float((test_pred == b_y.data.numpy()).astype(int).sum())
        total+=n
        # print(n, total, deno)
        if step % 100 == 0:
            # print(step)
            pass
            # print('Step: ', step, '| test loss: %.6f' % loss.data.numpy())
    print('EPOCH: ', epoch)
    print("[predictions/labels]=[{0}/{1}]".format(total, 10000))
    accuracy = total / float(10000)
    print('test accuracy: %.6f' % accuracy)

