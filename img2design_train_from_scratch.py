import os, errno
from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from img2design_dataset import Img2DesignDataset


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


expr_name = "controlnet_bsz_36_encode_with_vae"
wandb_logger = WandbLogger(project='img2design', name=expr_name)

# Configs
data_path = '/data/text2design/preprocess/img2design/theme_v1.0'
resume_path = './models/control_sd15_unet_with_vae_ini.ckpt'
output_dir = f'/home/v-binyanxu/ControlNet/model_ckpts/{expr_name}'
makedirs(output_dir)

batch_size = 6
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_steps=10000
accumulate_grad_batches=6

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
train_dataset = Img2DesignDataset(data_path, 'train')
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
eval_dataset = Img2DesignDataset(data_path, 'val')
eval_dataloader = DataLoader(eval_dataset, num_workers=0, batch_size=batch_size, shuffle=True)

img_logger_dir = os.path.join(output_dir, 'image_logs')
makedirs(img_logger_dir)
img_logger = ImageLogger(
    batch_frequency=logger_freq,
    local_save_dir=img_logger_dir
)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=output_dir,
    save_top_k=-1,
    every_n_train_steps=2000,
)
trainer = pl.Trainer(
    gpus=1, 
    precision=32, 
    callbacks=[img_logger, checkpoint_callback],
    accelerator="gpu",
    strategy="ddp",
    max_steps=max_steps,
    accumulate_grad_batches=accumulate_grad_batches,
    gradient_clip_val=0.5,
    logger=wandb_logger
)

# Train!
trainer.fit(model, train_dataloader, eval_dataloader)
