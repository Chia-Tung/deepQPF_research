import os
import sys
import yaml
import argparse
import torch
import torchvision

from src.data_manager import DataManager
from src.model_manager import ModelManager

os.environ['CUDA_VISIBLE_DEVICES']="0"
print('Python version', sys.version)
print('CudaToolKit Version', torch.version.cuda)
print('torch Version', torch.__version__)
print('torchvision Version', torchvision.__version__)

def main():
    config_file = './exp_config/exp1.yml'
    with open(config_file, "r") as content:
        config = yaml.safe_load(content)

    dm = DataManager(
        data_meta_info=config['train_config']['data_meta_info'], 
        **config['train_config']['data_loader_params']
    )

if __name__ == '__main__':
    main()

# import argparse
# import os, sys
# import socket
# from datetime import datetime
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# os.environ['ROOT_DATA_DIR']='/work/dong1128/database'
# # os.environ['ROOT_DATA_DIR']='/bk2/handsomedong/DLRA_database'

# import torch
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torchvision
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
# from torch.utils.cpp_extension import CUDA_HOME
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from core.data_loader_type import DataLoaderType
# from core.enum import DataType
# from core.loss_type import BlockAggregationMode, LossType
# from core.model_type import ModelType
# from data_loaders.pl_data_loader_module import PLDataLoader
# from utils.run_utils import get_model, parse_date_end, parse_date_start, parse_dict

# def main(args):
#     workers = 8
#     input_shape = (120, 120)
#     loss_type = int(args.loss_kwargs.get('type', LossType.WeightedMAE))
#     loss_aggregation_mode = int(args.loss_kwargs.get('aggregation_mode', BlockAggregationMode.MAX))
#     loss_kernel_size = int(args.loss_kwargs.get('kernel_size', 5))
#     residual_loss = bool(int(args.loss_kwargs.get('residual_loss', 0)))
#     mixing_weight = float(args.loss_kwargs.get('w', 1))

#     loss_kwargs = {
#         'type': loss_type,
#         'aggregation_mode': loss_aggregation_mode,
#         'kernel_size': loss_kernel_size,
#         'residual_loss': residual_loss,
#         'w': mixing_weight
#     }
#     if loss_type in [LossType.SSIMBasedLoss, LossType.NormalizedSSIMBasedLoss]:
#         loss_kwargs['mae_w'] = float(args.loss_kwargs.get('mae_w', 0.1))
#         loss_kwargs['ssim_w'] = float(args.loss_kwargs.get('ssim_w', 0.02))

#     data_kwargs = {
#         'data_type': int(args.data_kwargs.get('type', DataType.RAIN+DataType.RADAR)),
#         'residual': bool(int(args.data_kwargs.get('residual', 0))),
#         'target_offset': int(args.data_kwargs.get('target_offset', 0)),
#         'target_len': int(args.data_kwargs.get('target_len', 3)),
#         'input_len': int(args.data_kwargs.get('input_len', 6)),
#         'hourly_data': bool(int(args.data_kwargs.get('hourly_data', 0))),
#         'hetero_data': bool(int(args.data_kwargs.get('hetero_data', 0))),
#         'sampling_rate': int(args.data_kwargs.get('sampling_rate', 5)),
#         'prior_dtype': int(args.data_kwargs.get('prior', DataType.NONEATALL)),
#         'random_std': int(args.data_kwargs.get('random_std', 0)),
#         'ith_grid': int(args.data_kwargs.get('ith_grid', -1)),
#         'pad_grid': int(args.data_kwargs.get('pad_grid', 10)),
#         'threshold': float(args.data_kwargs.get('threshold', 0.5)),
#     }
#     model_kwargs = {
#         'adv_w': float(args.model_kwargs.get('adv_w', 0.01)),
#         'model_type': ModelType.from_name(args.model_kwargs.get('type', 'BalancedGRUAdverserialAttention')),
#         'dis_d': int(args.model_kwargs.get('dis_d', 3)),
#         'teach_force':float(args.model_kwargs.get('teach_force', 0)),
#     }

#     dm = PLDataLoader(
#         args.train_start,
#         args.train_end,
#         args.val_start,
#         args.val_end,
#         img_size=input_shape,
#         dloader_type=args.dloader_type,
#         **data_kwargs,
#         batch_size=args.batch_size,
#         num_workers=workers,
#     )

#     model = get_model(
#         args.train_start,
#         args.train_end,
#         model_kwargs,
#         loss_kwargs,
#         data_kwargs,
#         args.checkpoints_path,
#         args.log_dir,
#         data_loader_info=dm.model_related_info,
#     )
    
#     logger = TensorBoardLogger(save_dir='logs', name=ModelType.name(model_kwargs['model_type']))
#     logger.log_hyperparams(args)
#     checkpoint_callback = model.get_checkpoint_callback()
#     trainer = Trainer.from_argparse_args(
#         args, 
#         gpus=1,
#         max_epochs=50, 
#         fast_dev_run=False, 
#         logger=logger,
#         callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=5)],
#     )
#     trainer.fit(model, dm)  #.fit同時做了train and validation
#     #default max epochs for pl is 1000

# if __name__ == '__main__':
#     # original cmd from ASHESH
#     # python scripts/pl_run.py --train_start=20150101 --train_end=20150131 --val_start=20150201 --val_end=20150331 --gpus=1 --batch_size=2 --loss_kwargs=type:1,kernel:10,aggregation_mode:1 --data_kwargs=sampling_rate:3,hetero_data:0 --precision=16 --model_type=BaselineCNN
    
#     # training seasonal data as input
#     # python pl_run.py --batch_size=32 --data_kwargs=type:49,sampling_rate:3,target_len:3 --model_kwargs=type:BalancedGRUAdvPONIAtten,teach_force:0.5
#     # python pl_run.py --batch_size=32 --data_kwargs=type:49,sampling_rate:3,target_len:3 --model_kwargs=type:BalGRUAdvPONIAtten_addponi,teach_force:0.5
    
#     # training terrain as input
#     # python pl_run.py --batch_size=32 --data_kwargs=type:49,sampling_rate:3,target_len:3 --model_kwargs=type:BalancedGRUAdvPONI,teach_force:0.5
   
#     # train all data
#     # python pl_run.py --batch_size=32 --data_kwargs=type:59,sampling_rate:3,target_len:3 --model_kwargs=type:BalGRUAdvPONI_addponi,teach_force:0.5
#     print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
#     print('Python version', sys.version)
#     print('CUDA_HOME', CUDA_HOME)
#     print('CudaToolKit Version', torch.version.cuda)
#     print('torch Version', torch.__version__)
#     print('torchvision Version', torchvision.__version__)

#     parser = argparse.ArgumentParser() #--代表是optional
#     # 若在命令列有輸入值，才會進行「type」的運算（預設string）；若無，直接回傳default
#     parser.add_argument('--dloader_type', type=DataLoaderType.from_name, default=DataLoaderType.Native)
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--train_start', type=parse_date_start, default=datetime(2015, 1, 1))
#     parser.add_argument('--train_end', type=parse_date_end, default=datetime(2018, 12, 31, 23, 50))
#     parser.add_argument('--val_start', type=parse_date_start, default=datetime(2019, 1, 1))
#     parser.add_argument('--val_end', type=parse_date_end, default=datetime(2021, 12, 31, 23, 50))
#     parser.add_argument('--loss_kwargs', type=parse_dict, default={})
#     parser.add_argument('--log_dir', type=str, default='logs')
#     parser.add_argument('--data_kwargs', type=parse_dict, default={})
#     parser.add_argument('--model_kwargs', type=parse_dict, default={})
#     parser.add_argument('--checkpoints_path',
#                         type=str,
#                         default=(os.getcwd() + '/checkpoints/'),
#                         help='Full path to the directory where model checkpoints are [to be] saved')
#     # 加入Trainer參數
#     parser = Trainer.add_argparse_args(parser)
#     argm = parser.parse_args()
   
#     main(argm)
