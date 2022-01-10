import os
import shutil


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from torch import nn
import torchvision

from torchinfo import summary
# import torch.utils.tensorboard as tb

import metrics
import mnist
import cifar10

np.random.seed(0)
torch.manual_seed(10);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')