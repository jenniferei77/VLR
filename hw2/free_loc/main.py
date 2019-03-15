import argparse
import os
import shutil
import time
from datetime import datetime
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid
        
from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', default=True, action='store_true')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    #logdir = os.path.join('./main_output',
    #                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #if os.path.exists(logdir):
    #    shutil.rmtree(logdir)
    #os.makedirs(logdir)
 
    if args.vis:
        import visdom
        from tensorboardX import SummaryWriter
        vis = visdom.Visdom(server='http://localhost',port='8097')
        training_loss = vis.line(Y=np.array([0.8]), X=np.array([0.0]), opts=dict(title='Training Loss Curve', width=300, height=300, showlegend=False, xlabel='Global Step', ylabel='Loss'))
    #    print(logdir)
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter('/data/VLR/hw2/main_output/exp1' + date_time)
    else:
        vis = None
        writer = None

    classes = np.asarray(train_dataset.classes)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, vis, training_loss, writer)
        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, vis, writer, epoch, classes)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, vis, training_loss, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    #Create global step for vis graph
    global_step = []

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        output = model(input_var)
        #print('Train Model Output Shape: ', output.shape)

        maxpool = nn.MaxPool2d(kernel_size=output.shape[-1], stride=1)
        imoutput = maxpool(output)  
        imoutput = imoutput.view(imoutput.size(0), -1)
        #print('Train Maxpool Output Shape: ', imoutput.shape)
        #print('Target Shape: ', target_var.long().shape)

        #loss = criterion(imoutput, torch.autograd.Varible(target_var.max(dim=1)[1]).long())
        loss = criterion(imoutput, target_var)
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        #m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1, input.size(0))
        #avg_m2.update(m2, input.size(0))

        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1))
            #print('Epoch: [{0}][{1}/{2}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
            #      'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
            #          epoch,
            #          i,
            #          len(train_loader),
            #          batch_time=batch_time,
            #          data_time=data_time,
            #          loss=losses,
            #          avg_m1=avg_m1,
            #          avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        #writer.add_scalar('train/loss', loss.item(), )
        global_step.append(i+(len(train_loader)*epoch))
        if epoch == 1 and i == 0:
            writer.add_scalar('train/loss', loss.item(), np.array([i]))
            vis.line(Y=np.array([loss.item()]), X=np.array([i]), opts=dict(title='Training Loss Curve', width=300, height=300, update='append'))
        else:
            writer.add_scalar('train/loss', loss.item(), np.array([i+(len(train_loader)*epoch)]))
            vis.line(Y=np.array([loss.item()]), X=np.array([i+(len(train_loader)*epoch)]), win=training_loss, update='append')

        # End of train()


def validate(val_loader, model, criterion, vis, writer, epoch, classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    image_freq = int(len(val_loader) / 4)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        output = model(input_var)
        maxpool = nn.MaxPool2d(kernel_size=output.shape[-1], stride=1)
        imoutput = maxpool(output)
        imoutput = imoutput.view(imoutput.size(0), -1)
        loss = criterion(imoutput, target_var)
        #loss = criterion(imoutput, torch.autograd.Varible(target_var.max(dim=1)[1]).long())
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        #m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1, input.size(0))
        #avg_m2.update(m2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1))
            #print('Test: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
            #      'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
            #          i,
            #          len(val_loader),
            #          batch_time=batch_time,
            #          loss=losses,
            #          avg_m1=avg_m1,
            #          avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        writer.add_scalar('train/loss', loss.item(), i)
        if i % image_freq:
            images = input_var[0:2]
            targets = target_var[0:2]
            class_heat1 = np.argmax(targets[0])
            class_heat2 = np.argmax(targets[1])
            image_title = 'epoch' + str(epoch) + '_iter' + str(i) + '_batchInds1&2_image'
            writer.add_images(image_title, images)
            heatmap_title1 = 'epoch' + str(epoch) + '_iter' + str(i) + '_batchInd1_heatmap_' + classes[class_heat1]
            heatmap_title2 = 'epoch' + str(epoch) + '_iter' + str(i) + '_batchInd2_heatmap_' + classes[class_heat2]
            vis.images(images, opts=dict(title=image_title))
            vis.heatmap(output[0][class_heat1], opts=dict(title=heatmap_title1, colormap='Electric'))
            vis.heatmap(output[1][class_heat2], opts=dict(title=heatmap_title2, colormap='Electric'))
            heatmap_title = 'epoch' + str(epoch) + '_iter' + str(i) + '_batchInd1&2_heatmap_' + classes[class_heat1] + '&' + classes[class_heat2]
            
            output_to_show = torch.stack((output[0][class_heat1], output[1][class_heat2]))
            output_grid = make_grid(output_to_show)
            writer.add_image(heatmap_title, output_to_show)

    #print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
    #    avg_m1=avg_m1, avg_m2=avg_m2))

    #return avg_m1.avg, avg_m2.avg
    return avg_m1.avg, 1.0

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    AP = []
    for single_class in range(output.shape[-1]):
        predicts = output[:,single_class]
        targets = target[:,single_class]
        _, sorted_inds = torch.sort(predicts, 0, True)
        sorted_targs = np.asarray(targets[sorted_inds])
        sorted_preds = np.asarray(predicts[sorted_inds].cpu().detach().numpy())
        class_ap = sklearn.metrics.average_precision_score(sorted_targs, sorted_preds)
        AP.append(class_ap)

    mAP = np.nanmean(AP)
    return mAP


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    AP = []
    for single_class in range(output.shape[-1]):
        predicts = output[:,single_class]
 

    return [1]


if __name__ == '__main__':
    main()
