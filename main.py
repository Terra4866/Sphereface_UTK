import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import dataset
from utils import AverageMeter, Logger
import net_sphere
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch sphereface for UTK')
# dataset
parser.add_argument('-d', '--dataset', type=str, default='UTK')
parser.add_argument('-j', '--workers', default=4, type=int)
# optimization
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=20)

# model
parser.add_argument('--model', type=str, default='sphere20a')
parser.add_argument('--label', type=str, default = 'gender')
# misc
parser.add_argument('--gpu', type=str, default='0, 1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--print_freq', type = int, default = 80)

args = parser.parse_args()

if args.label == 'age':
    num_class = 117
elif args.label == 'gender':
    num_class = 2
elif args.label == 'race':
    num_class = 5

model_PATH = "model/" + args.label + "_" + args.model + ".pth"

TRAIN_LAND_PATH = "landmark/landmark_1_2.txt"
TRAIN_ROOT_PATH = 'data/data/UTKFace'

TEST_LAND_PATH = "/landmark/landmark_list_part3.txt"
TEST_ROOT_PATH = 'data/data/UTKFace'


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False


    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))


    trans = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_set = dataset.UTK_DS(TRAIN_LAND_PATH,TRAIN_ROOT_PATH, args.label, trans) 

    test_set = dataset.UTK_DS(TEST_LAND_PATH,TEST_ROOT_PATH, args.label, trans)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size = args.bs, shuffle = True,  num_workers = args.workers, drop_last = True)

    testloader = torch.utils.data.DataLoader(test_set, batch_size = args.bs, shuffle = True,  num_workers = args.workers, drop_last = True)

    print("Creating model: {}".format(args.model))
    if args.model == 'sphere20a':
        model = net_sphere.sphere20a(classnum = num_class)   
    else:
        model = net_sphere.sphere64a(classnum = num_class)  

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if os.path.isfile(model_PATH):
        print("\nLoading the lastest model\n")
        model.load_state_dict(torch.load(model_PATH))
        model.eval()

    criterion = net_sphere.AngleLoss()

    lr = args.lr


    start_time = time.time()

    for epoch in range(args.max_epoch):
        if epoch > 0 and epoch % == 0:
            lr = lr * 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.5, weight_decay = 5e-4)


        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion,
              optimizer, trainloader, use_gpu, num_class, epoch)

        
        print("==> Test")
        acc, err = test(model, testloader, use_gpu, num_class , epoch)
        print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    torch.save(model.state_dict(), model_PATH)

def train(model, criterion, optimizer, trainloader, use_gpu, num_classes, epoch):
    model.train()

    losses = AverageMeter()

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
           data, labels = data.cuda(), labels.cuda()
        #data, labels = Variable(data), Variable(labels)
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} " \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))



def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0


    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            outputs = outputs[0]
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            if batch_idx % 1000 == 0:
                 print("predictions : {} \n     answer : {}\n".format(predictions, labels.data))
            
    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()





