img_wavelets = False ## by @dv if you need WT
swt = True
#!/user/bin/python
# coding=utf-8
import os
from os.path import join,  isdir, dirname,  abspath  #split, splitext, isfile
from pathlib import Path

#from utils import Logger, arg_parser
if (swt):
    from modules.data_loader import Dataset_s3_swt as Dataset_s3
elif (img_wavelets):
    from modules.data_loader import Dataset_s3_w as Dataset_s3 ## original, commented by @dv ## use when wavelet of input image is required
else:
    from modules.data_loader import Dataset_s3 ## added by @dv ## use when WT is not required
from modules.data_loader import  S3Images
#from modules.models import HED
from msnet import msNet
from modules.trainer import Network, Trainer
from modules.utils import struct
from modules.transforms import Fliplr, Rescale_byrate
#from modules.options import arg_parser

import torch.cuda
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime

from pandas import read_csv


root=Path(".")#macgregor', 'image')
tag = datetime.now().strftime("%y%m%d-%H%M%S")+'-DWT-HED-VGG16-4SOs-maryam-3-iswt'


params={
     'root': root,
     'tmp': Path(f'../model-outputs/tmp/iceIBK-{tag}'), ##os.getcwd()
     'log_dir': Path(f'../model-outputs/logs/iceIBK-{tag}'),
     'dev_dir':root/'sample36',
     'val_percent': 0,
     # 'start_epoch' : 7, # by @dv
     'start_epoch' : 0,
     'max_epoch' : 15,
     'batch_size': 1,
     'itersize': 10,
     'stepsize': 3,
     'lr': 1e-06,
     'momentum': 0.9,
     'weight_decay': 0.0002,
     'gamma': 0.1,
     'pretrained_path': None,
     # 'resume_path': 'G:/My Drive/Research_Debvrat/Codes/Wavelet/cresis_sr/model-outputs/tmp/iceIBK-211118-181924-HED-VGG16-SoWT1-dmey/checkpoint_epoch7.pth', # by @dv
     'resume_path': None,
     'use_cuda': torch.cuda.is_available()
     }

args= struct(**params)


#%% def main():

if not isdir(args.tmp):
    os.makedirs(args.tmp)
#%% define network
net=Network(args, model=msNet())


#%% train dataset S3
si=S3Images('aws_access_key_id', 'aws_secret_access_key')

# df_train=read_csv(root/'lst_files/2012_dry_train.lst', header=None)
df_train=read_csv(root/'lst_files/frames_001_243_20120330_04_dv_train.lst', header=None)

# images=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_train[0].values] # for lora's 2012 data
images=['Debvrat/frames_001_243_20120330_04_dv/'+item for item in df_train[0].values]
# images=['Debvrat/2012_dry/'+item for item in df_train[0].values]

# ds=Dataset_s3(bucket='cresis', keys=images, s3Get=si.from_s3)
# train_loader= DataLoader(ds, batch_size=1, shuffle=True) ## by @dv

# bucket = 'cresis' # 'cresis' or 'binalab-users' added by @dv
bucket = 'binalab-users' # 'cresis' or 'binalab-users' added by @dv

ds=[Dataset_s3(bucket=bucket, keys=images, s3Get=si.from_s3),
Dataset_s3(bucket=bucket, keys=images, s3Get=si.from_s3, transform=Rescale_byrate(.75)),
Dataset_s3(bucket=bucket, keys=images, s3Get=si.from_s3,transform=Rescale_byrate(.5)),
Dataset_s3(bucket=bucket, keys=images, s3Get=si.from_s3,transform=Rescale_byrate(.25)),
Dataset_s3(bucket=bucket, keys=images, s3Get=si.from_s3, transform=Fliplr())
] ### commented by @dv
# Loader
train_dataset=ConcatDataset(ds) ## original, commented by @dv
train_loader= DataLoader(train_dataset, batch_size=1, shuffle=True) ## original, commented by @dv


# df_dev=read_csv(root/'dev_s3.lst', header=None)
# images_dev=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_dev[0].values]

# dev_dataset=Dataset_s3(bucket='cresis', keys=images_dev, s3Get=si.from_s3) ## commented by @dv
# dev_loader= DataLoader(dev_dataset, batch_size=1) ## commented by @dv

#%% train dataset
# ds=[Dataset_ls(root=root,lst='train_pair.lst'),
# Dataset_ls(root=root,lst='train_pair.lst', transform=Rescale_byrate(.75)),
# Dataset_ls(root=root,lst='train_pair.lst',transform=Rescale_byrate(.5)),
# Dataset_ls(root=root,lst='train_pair.lst',transform=Rescale_byrate(.25)),
# Dataset_ls(root=root,lst='train_pair.lst', transform=Fliplr())
# ]
#train_dataset=Dataset_ls(root=root,lst='train_pair.lst')


# dev dataset optional
#dev_dataset=Dataset_ls(root=root,lst='dev.lst')
#dev_loader= DataLoader(dev_dataset, batch_size=1)

#%% define trainer
trainer=Trainer(args,net, train_loader=train_loader)
#%%

startTime = datetime.now() ## @dv
# switch to train mode: not needed!  model.train()
for epoch in range(args.start_epoch, args.max_epoch):

    ## initial log (optional:sample36)
    # if epoch == 0: ## commented by @dv
    #     print("Performing initial testing...") ## commented by @dv
    #     trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'), epoch=epoch) ## commented by @dv

    ## training
    trainer.train(save_dir = args.tmp, epoch=epoch)

    ## dev check (optional:sample36)
    # trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-epoch-%d' % epoch), epoch=epoch) ## commented by @dv

endTime = datetime.now() ## @dv
print('\n\n Training complete. Output directory: ' + tag)
print(startTime.strftime("%y%m%d-%H%M%S"))
print(endTime.strftime("%y%m%d-%H%M%S"))
# if __name__ == '__main__':
#     main()
