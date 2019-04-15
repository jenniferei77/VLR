from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer
from test import *
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
import gc
import subprocess
import pdb


try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 500

start_step = 0
end_step = 30000
lr_decay_steps = {150000}
lr_decay = 1. / 10
thresh = 0.001

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False

remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load imdb and create data later
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load test imdb and create data layer
test_imdb = get_imdb('voc_2007_test')
#test_imdb.competition_mode(on=True)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG, training=True)
print(net)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue
own_state['classifier.0.weight'].copy_(pret_net['classifier.1.weight'].data)
own_state['classifier.0.bias'].copy_(pret_net['classifier.1.bias'].data)
own_state['classifier.3.weight'].copy_(pret_net['classifier.4.weight'].data)
own_state['classifier.3.bias'].copy_(pret_net['classifier.4.bias'].data)

#own_state['fc6.fc.weight'].copy_(pret_net['classifier.1.weight'])
#own_state['fc6.fc.bias'].copy_(pret_net['classifier.1.bias'])
#own_state['fc7.fc.weight'].copy_(pret_net['classifier.4.weight'])
#own_state['fc7.fc.bias'].copy_(pret_net['classifier.4.bias'])



# Move model to GPU and set train mode
#net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
sgd_params = list(net.parameters())[2:]
optimizer = torch.optim.SGD(sgd_params, lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
test_name = os.path.join(output_dir, '_eval', date_time)
os.makedirs(test_name)
if use_tensorboard:
    from tensorboardX import SummaryWriter
    writer_path = os.path.join(output_dir, '_tensorboard', date_time)
    os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

if use_visdom:
    import visdom
    vis = visdom.Visdom(server='http://localhost', port='8097')
    training_loss = vis.line(Y=np.array([0.8]), X=np.array([0.0]), opts=dict(title='Training Loss Curve', width=300, height=300, showlegend=False, xlabel='Global Step', ylabel='Loss'))
    testing_mAP = vis.line(Y=np.array([0.8]), X=np.array([0.0]), opts=dict(title='Testing mAP', width=300, height=300, showlegend=False, xlabel='Global Step', ylabel='Test mAP'))

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
mAP = None
for step in range(start_step, end_step + 1):

    net.train() 
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']
    
    # forward
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.item()
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps, lr,
            momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every N iterations (N defined in handout)
    if step % 5000 == 0:
        net.eval()
        test_imdb.competition_mode(on=True)
        aps = test_net(test_name + '_step_' + str(step), net, test_imdb, max_per_image=300, thresh=0.0001, visualize=True, logger=writer, step=step)
        mAP = np.nanmean(np.asarray(aps))
    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
    #if visualize and step % vis_interval == 0:
    if visualize and step % 500 == 0:
        #TODO: Create required visualizations
        if use_tensorboard:
            print('Logging to Tensorboard')
            writer.add_scalar('train/loss', loss.item(), step)
            if step % 2000 == 0:
                hist_iter = 0
                for m in net.modules():
                    for name, param in m.named_parameters():
                        if 'classifier.3.weight' in name or 'features.3.weight' in name or 'features.0.weight' in name:
                            writer.add_histogram('weights_hist/layer' + str(hist_iter), param.data.cpu().numpy(), hist_iter)
                            writer.add_histogram('grad_hist/layer' + str(hist_iter), param.grad.data.cpu().numpy(), hist_iter)
                            hist_iter += 1
        if use_visdom:
            print('Logging to visdom')
            vis.line(Y=np.asarray([loss.item()]), X=np.array([step]), win=training_loss, update='append')
            if mAP != None and step % 5000 == 0:
                vis.line(Y=np.asarray([mAP]), X=np.array([step]), win=testing_mAP, update='append')


    # Save model occasionally
    #if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
    if (step % 15000 == 0) and step > 0:
        save_name = os.path.join(
            output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX, step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(
            optimizer.param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
