import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import time

from tensorboardX import SummaryWriter
import os
from utils import cutout,mixup, mixup_criterion, cutmix, cutmix_criterion
import numpy as np
    
def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #print("conv1 size：", out.size())
        out = self.layer1(out)
        #print("layer1 size：", out.size())
        out = self.layer2(out)
        #print("layer2 size：", out.size())
        out = self.layer3(out)
        #print("layer3 size：", out.size())
        out = self.layer4(out)
        #print("layer4 size：", out.size())
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("final layer size：", out.size())
        out = self.fc(out)
        return out
def ResNet18():
    return ResNet(ResidualBlock)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = 0
    num = 0
    # switch to train mode
    model.train()

    for i , data in enumerate(train_loader):
        
        #inputs, targets = cutout(data)
        if args.mixup:
            r = np.random.rand()
            if r < args.mixup_prob:
                inputs, targets, targets1, lam = mixup(data,beta=args.beta)
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets1 = targets.cuda()
                # forward
                output = model(inputs)
                loss = mixup_criterion(criterion,output, targets, targets1, lam)
            else:
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()

                # forward
                output = model(inputs)
                loss = criterion(output, targets)
        elif args.cutmix:
            r = np.random.rand()
            if r < args.cutmix_prob:
                inputs, targets, targets1, lam = cutmix(data,beta=args.beta)
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets1 = targets.cuda()
                # forward
                output = model(inputs)
                loss = cutmix_criterion(criterion,output, targets, targets1, lam)
            else:
                inputs, targets = data
                inputs = inputs.cuda()
                targets = targets.cuda()

                # forward
                output = model(inputs)
                loss = criterion(output, targets)
        elif args.cutout:
            r = np.random.rand()
            if r < args.cutout_prob:
                inputs, targets = cutout(data)
            else:
                inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()

            # forward
            output = model(inputs)
            loss = criterion(output, targets)
        else:
            inputs, targets = data
            inputs = inputs.cuda()
            targets = targets.cuda()

            # forward
            output = model(inputs)
            loss = criterion(output, targets)


        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses += loss.item() * inputs.size(0)
        num += inputs.size(0)

    log = 'Epoch:{0}\tTrain Loss: {loss:.4f}\t'.format(epoch, loss=losses/num)
    return log, losses/num
def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    # lr = initial_lr * (0.1 ** (epoch // (num_epochs * 0.5))) * (0.1 ** (epoch // (num_epochs * 0.75)))
    #lr = initial_lr * lr_decay ** int(epoch/decay_step)
    # lr = initial_lr * 1.0/ (1.0 + lr_decay*epoch)
    """decrease the learning rate at 50 and 100 epoch"""
    lr = initial_lr
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(test_loader, criterion, model):
    top1_acc = 0
    num = 0
    losses = 0

    # switch to evaluate mode
    model.eval()

    for inputs, targets in test_loader:

        # forward
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            
        # compute accuracy and loss
        acc1 = accuracy(output.data, targets)[0]
        top1_acc += acc1.item() * inputs.size(0)
        losses += loss.item() * inputs.size(0)
        num += inputs.size(0)

    log = 'Test Loss: {loss:.4f}'.format(loss=losses/num) + '\t'+ 'Test Acc@1: {top1:.3f}%'.format(top1=top1_acc/num)

    return top1_acc/num, log, losses/num

def main(args):
    model = ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.99))
    # optimizer =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)

    best_acc = 0
    best_epoch = 0
    best_state = 0
    start = time.time()
    
    writer = SummaryWriter(log_dir=args.logs_dir)
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.num_epochs)
        train_log, train_loss = train(cifar_train_loader, model, criterion, optimizer, epoch)
        acc, test_log, test_loss = test(cifar_test_loader, criterion, model)
        log = train_log + test_log
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            best_epoch = epoch
            best_state = model.state_dict()
        writer.add_scalar("train_loss", train_loss, epoch)    
        writer.add_scalar("test_loss", test_loss, epoch)    
        writer.add_scalar("test_acc", acc, epoch)    
        write_to_record_file(log,args.record_file)
    end = time.time()
    print(end - start)
    best_log = 'Best Test Acc@1: {top1:.4f}'.format(top1=best_acc) + '\t'+ 'Best Epoch: {epoch:d}'.format(epoch=best_epoch)
    write_to_record_file(best_log, args.record_file)
    torch.save({'epoch':best_epoch,'state_dict':best_state,'acc': best_acc,}, os.path.join(args.ckpt_dir, 'best_test'))

def setup_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    
    # std=[0.2023, 0.1994, 0.2010]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cutmix_aug_01', help='net type')
    parser.add_argument('-batch_size', type=int, default=512, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume',default=False, help='resume training')
    parser.add_argument('-num_epochs', default=200, help='num epoch')
    parser.add_argument('-ckpt_dir', type=str,default=None, help='ckpts')
    parser.add_argument('-logs_dir', type=str,default=None, help='logs')
    parser.add_argument('-record_file', type=str,default=None, help='record')
    parser.add_argument('-mixup', type=bool,default=False, help='record')
    parser.add_argument('-mixup_prob', type=float,default=0.5)
    parser.add_argument('-cutmix', type=bool,default=True, help='record')
    parser.add_argument('-cutmix_prob', type=float,default=0.1)
    parser.add_argument('-cutout', type=bool,default=False, help='record')
    parser.add_argument('-cutout_prob', type=float,default=0.5)
    parser.add_argument('-beta', type=float,default=1.0)
    args = parser.parse_args()
    
    # data dir
    cifar_data_path = '/remote-home/mfdu/nlpbeginner/hw2'
    # para
    args.ckpt_dir = os.path.join(args.name, 'ckpts')
    args.logs_dir = os.path.join(args.name, 'logs')
    os.makedirs(args.name, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    args.record_file = os.path.join(args.logs_dir, 'train.txt')
    
    # 设置随机数种子
    setup_seed(1234)
    normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    cifar_train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        normalize,
    ])
    cifar_test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    cifar_train = CIFAR100(cifar_data_path, download=True, transform=cifar_train_transform)
    cifar_test = CIFAR100(cifar_data_path, train=False, transform=cifar_test_transform)
    cifar_train_loader = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    main(args)
