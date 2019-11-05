'''
fashion mnist classification
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
from loss import compute_center_loss, get_center_delta
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
LAMBDA = 1
ALPHA = 0.01
DOWNLOAD_FASHIONMNIST = False

# load fashion mnist dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_FASHIONMNIST = True
if os.path.exists('accuracy.txt'):
    os.remove('accuracy.txt')
# DOWNLOAD_FASHIONMNIST = True
# load train data
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.Compose([#transforms.ToPILImage(),
                                  transforms.RandomRotation(15),
                                  RandomShift(0.3),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]), ## 有问题。
    download=DOWNLOAD_FASHIONMNIST
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
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
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:500]
# test_y = test_data.test_labels[:500]
# print(test_x.data.numpy().shape)
# print(test_data.test_labels.data.numpy().shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        feature_dim=512
        self.register_buffer('centers', (
                torch.rand(10, feature_dim).cuda() - 0.5) * 2)
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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.extract=nn.Sequential(
            # nn.Dropout(p=0.25),
            nn.Linear(128*7*7, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.25),
            nn.Linear(512, 512),)
        self.classifier=nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.25),
            nn.Linear(512, 10),)
            #nn.Softmax(dim=0))
 
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
        features = self.extract(x)
        x = self.classifier(features)
        # print(x.shape)
        return x, features

def mnist_vis(b_x, b_y):
    b_x, b_y = np.squeeze(b_x.data.numpy()), np.squeeze(b_y.data.numpy())
    plt.figure()
    print(b_x.shape)
    for i in range(4):
        for j in range(6):
            plt.subplot(4,6,i*6+j+1)
            print(i*6+j)
            plt.title(b_y[i*6+j])
            plt.imshow(b_x[i*6+j, :, :])
            # print(b_x)
            plt.axis('off')
        # print(i)
    plt.axis('off')
    plt.savefig('data_vis.jpg')
    plt.show()

def ohkpm(loss, top_k, f=1):
 
        ohkpm_loss = 0.
        topk_val, topk_idx = torch.topk(f*loss, k=top_k, dim=0, sorted=False)
        tmp_loss = torch.gather(loss, 0, topk_idx)
        # print(tmp_loss.shape)
        ohkpm_loss = torch.unsqueeze(tmp_loss, dim=0).mean()
        # print(ohkpm)
        # ohkpm_loss = torch.cat(ohkpm_loss_list, 0).mean()
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

    reload = False
    best_acc = 0.
    # from resnet_mnist import resnet_mnist
    # cnn = resnet_mnist().cuda()
    cnn = CNN().cuda()
    epoch = 0
    loss_avg = AverageMeter()
    # cnn = CNN().cuda()
    print(cnn)
    if reload:
        model = torch.load('mnist_best.pth')
        cnn.load_state_dict(model['state_dict'])
        epoch = model['epoch']
        best_acc = model['acc']

    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(params=cnn.parameters(), lr=LR)
    # optimizer = torch.optim.RMSprop(params=cnn.parameters, lr=LR)
    # loss_func = nn.CrossEntropyLoss(size_average=False, reduce=False)
    loss_func = nn.CrossEntropyLoss()
    
    total = 0.

    while epoch < EPOCH:
        if epoch in [35, 45]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR*0.1
        cnn.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            # mnist_vis(b_x, b_y)
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            predictions, features = cnn(b_x)
            cel = loss_func(predictions, b_y)
            centers = cnn.centers
            # print(loss)
            # if epoch > 160:
            # cel = ohkpm(cel, 24)
            center_loss = compute_center_loss(features, centers, b_y)
            loss = LAMBDA * center_loss + cel
            center_deltas = get_center_delta(features.data, centers, b_y, ALPHA)
            cnn.centers = centers-center_deltas
            # else:
                # loss = ohkpm(loss, 24)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss = loss.cpu()
            # loss_avg.val = loss.data.numpy()
            if step % 500 == 0:
                print('Step: ', step, '| class loss: %.8f' % cel, step, '| center loss: %.8f' % center_loss)
        cnn.eval()
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            test_pred, _ = cnn(b_x)
            test_pred = test_pred.cpu()
            test_pred = torch.max(test_pred, 1)[1].data.squeeze().numpy()

            b_x = b_x.cpu()
            b_y = b_y.cpu()
            n = float((test_pred == b_y.data.numpy()).astype(int).sum())
            total+=n
            # print(n, total, deno)
            if step % 100 == 0:
                print(step)
                # print('Step: ', step, '| test loss: %.6f' % loss.data.numpy())
        print('EPOCH: ', epoch)
        print("[predictions/labels]=[{0}/{1}]".format(total, 10000))
        accuracy = total / float(10000)
        print('test accuracy: %.6f' % accuracy)
        # with open('accuracy.txt', 'a') as file:
            # file.write(str(loss_avg.avg) + ' ' + str(accuracy)+'\n')
        if accuracy > best_acc:
            best_acc = accuracy
            print("best: True")
            torch.save({'epoch': epoch, 'state_dict':cnn.state_dict(), "acc":best_acc}, "mnist_best.pth")
        print("best_acc: {0}".format(best_acc))
        total = 0
        epoch+=1
