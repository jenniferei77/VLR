import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn

import torchvision.models as models

from datasets.factory import get_imdb
from custom import *

import pdb

def main():
    imdb = get_imdb('voc_2007_trainval')
    classes, class_to_idx = find_classes(imdb)
    imgs = make_dataset(imdb, class_to_idx)
    pdb.set_trace() 

if __name__ == '__main__':
    main()
