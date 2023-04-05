import os, errno
from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from img2design_dataset import Img2DesignDataset


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
state_dict = load_state_dict('/home/v-binyanxu/ControlNet/ControlNet/models/v1-5-pruned.ckpt', location='cpu')
for key in state_dict.keys():
    if key.startswith('model.diffusion_model'):
        print(key)