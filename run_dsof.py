import argparse
import os
import torch
import random
import numpy as np
import datetime
import importlib
import yaml
from utils.misc import bcolors
from utils.tools import DataLogger
from dataloader import DatasetCustom 
from torch.utils.data import DataLoader
import pandas as pd
def set_seed(seed: int):
    """
    设置所有随机数生成器的种子以确保结果可复现
    
    Args:
        seed: 随机数种子值
    """
    # Python内置随机模块
    random.seed(seed)
    
    # NumPy随机模块
    np.random.seed(seed)
    
    # PyTorch随机模块
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
        # 设置CuDNN确定性操作（可能会降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"设置随机种子为: {seed}")
class MergeArguments:
    def __init__(self,  args, parser):
        self.parser = parser
        self.specified_program_options = args.__dict__.copy()

        self.sanity_check(args)
        
    def sanity_check(self, args):
        # Ensure all required arguments are provided
        assert args.data is not None, 'Data is not defined'
        assert args.y_model_main is not None, 'Main model is not defined'
        assert args.y_opt is not None, 'optimizer is not defined'
        # student model is optional
        assert args.y_trainer is not None, 'Trainer is not defined'
        
        if args.use_gpu: assert torch.cuda.is_available(), 'GPU is not available'

    def get_subargs(self, path):
        # Load sub-arguments from a YAML file
        with open(path) as f:
            subargs = yaml.load(f, Loader=yaml.FullLoader)    
        return subargs
        
    def add_subargs(self, path, header = None):
        # Add sub-arguments to the parser
        if not os.path.exists(path):
            print(f"{bcolors.WARNING}Warning: File {path} does not exist {bcolors.ENDC}")
            return 
        
        subargs = self.get_subargs(path)
        if header: subargs = {header: subargs}
        self.parser.set_defaults(**subargs)    
    
    def parse_args(self):
        # Parse the arguments with the specified defaults
        self.parser.set_defaults(**self.specified_program_options)
        return self.parser.parse_args()


class PostprocessArguments:
    
    def set_multi_gpu(self, args):
        # TODO: implement this
        # if args.use_gpu and args.use_multi_gpu:
        #     args.devices = args.devices.replace(' ','')
        #     device_ids = args.devices.split(',')
        #     args.device_ids = [int(id_) for id_ in device_ids]
        #     args.gpu = args.device_ids[0]
        
        return args

    def set_dataset_specs(self, args):
        # Set dataset specifications based on the features
        args.enc_in, args.dec_in, args.c_out = map(lambda dd: dd[args.features], [args.enc_in, args.dec_in, args.c_out])
        return args
    
    def set_settings(self, args, exp_starttime, itr):
        # Set experiment settings based on the provided arguments
        if args.test_run: 
            args.setting = f'logs/test_run/itr{itr}'
            return args
    
        get_header = lambda x: x.split('/')[0]
        replace_slash = lambda x: x.replace('/', '_')
        
        y_model_student = replace_slash(args.y_model_student) 
        
        if len(y_model_student.split('_') ) > 2:
            y_model_student = '_'.join(y_model_student.split('_')[:2])
            
        try:
            int(args.y_opt.split('/')[0].split('_')[-1])
            lr = "_"+args.y_opt.split('/')[0].split('_')[-1]
            
        except ValueError:
            lr = ""
        
        args.setting = f"exps/{get_header(args.y_model_main)}-{get_header(args.y_model_student)}-{get_header(args.y_trainer)}/"
        args.setting += f"{replace_slash(args.y_model_main)}--{y_model_student}--{replace_slash(args.y_trainer)}{lr}/"
        args.setting += f"{args.data}_pl{args.pred_len}/{exp_starttime}/itr{itr}" 
        
        return args

    def set_device(self, args):
        # Set the device for computation (CPU/GPU)
        args.device = DeviceSettings().get_cuda_device(args.gpu_id)
        return args

class DeviceSettings:
    def threading(self, max_threads = None):
        # Set the number of threads for computation
        if max_threads is None: return 
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
    
    def seeding(self, seed):
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def get_cuda_device(self, gpu_id):
        # Get the CUDA device based on the GPU ID
        return torch.device(f'cuda:{gpu_id}')
        
    def cuda_backends(self, use_cudnn = True, deterministic = False, benchmark = False, use_tf32 = False):
        # Configure CUDA backends
        torch.backends.cudnn.enabled = use_cudnn
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32

class LoadData:
    @staticmethod
    def _get_data( args, flag):
        # Load data based on the flag (train/val/test)
        timeenc = args.timeenc

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz #if args.online_learning != 'none' else args.batch_size;
            freq = args.freq[-1:]
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.freq
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq[-1:]

        data_set = DatasetCustom(
            root_path=args.root_path,
            data= args.data,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len if not hasattr(args, 'ar_pred_len') else args.ar_pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            online='full',
        )
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    
    
def get_total_params(model):
    # Get the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
import math
def save_hparams(args):
    # Save hyperparameters to a YAML file
    hparams = filter(lambda x: x[0] not in  ['gpu', 'use_multi_gpu', 'devices', 'use_gpu'], 
                args.__dict__.items())
    with open(os.path.join(args.setting, '../hparams.yaml'), 'w+') as file:
        yaml.dump(dict(hparams), file)

def train_baseline(main_model,data,seed,pred_len,
                   student_model='MLP',opt='w_student',trainer='w_student/residual/dsof'):
    
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
    # main_model = "DLinear"
    # student_model = "MLP"
    # opt = "w_student"
    # trainer = "w_student/residual/dsof"
    # MAIN settings
    parser.add_argument('--data', type=str, default='ETTh2', help='data of experiment')
    parser.add_argument('--data_path', type=str, default='ETTh2/ETTh2.csv', help='data of experiment')
    parser.add_argument('--root_path', type=str, default='dataset/', help='data of experiment')
    parser.add_argument('--y_model_main', type=str, default='iTransformer', help='main model of experiment')
    parser.add_argument('--y_model_student', type=str, default='MLP', help='student model of experiment')
    parser.add_argument('--y_opt', type=str, default='w_student', help='main optimizer of experiment')
    parser.add_argument('--y_trainer', type=str, default='w_student/residual/dsof', help='trainer of experiment')
    parser.add_argument('--comments', type=str, default='test',help='exp description')

    # general settings
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--test_run', action='store_true', default=False, help='test run the code')
    # parser.add_argument('--timing', action='store_true', default=False, help='test run the code')
    parser.add_argument('--resume', type=str, default=None, help='specify the model path and resume its training')

    # common dataset and dataloader settings
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # devices
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    # parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')


    ###########################################################################################

    # Parse the initial arguments
    args = parser.parse_args()
    args.data=data
    args.y_model_main=main_model
    args.seed=seed
    set_seed(args.seed)
    args.pred_len=pred_len
    args.data_path=fr'{args.data}/{args.data}.csv'
    # Initialize the MergeArguments class with the parsed arguments and parser
    merger = MergeArguments(args, parser)
    args.device = 'cuda:1'


    # Define the paths to the YAML configuration files for data, main model, optimizer, and trainer
    subargs = {
        f'./configs/data/{args.data}.yaml': None,
        f'./configs/model/{args.y_model_main}.yaml': 'main_model',
        f'./configs/optimizer/{args.y_opt}/{args.y_model_main}_{args.y_model_student}/{args.data}.yaml': None,
        f'./configs/trainer/{args.y_trainer}.yaml': None
    }

    # Add the sub-arguments from the YAML files to the parser
    for path, header in subargs.items():
        merger.add_subargs(path, header)

    # Define the path to the YAML configuration file for the student model
    subargs2 = {
        f'./configs/model/{args.y_model_student}.yaml': 'student_model',
    }

    # Add the sub-arguments from the student model YAML file to the parser
    for path, header in subargs2.items():
        merger.add_subargs(path, header)

    # Parse the arguments again with the updated defaults
    args = merger.parse_args()
    if args.y_model_main=='xPatch':
        args.learning_rate=0.0001
        args.batch_size=128
        args.train_epochs=20
        args.patch_len=8
        args.stride=4
    if args.y_model_main=='Leddam':
        args.train_epochs=20
        args.d_model=256
        args.dropout=0
        args.learning_rate=0.0001

    ###########################################################################################

    # Get the current time for experiment start time
    exp_starttime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Set dataset specifications based on the features
    args = PostprocessArguments().set_dataset_specs(args)

    # Set the device for computation (CPU/GPU)
    args = PostprocessArguments().set_device(args)

    # Configure threading and CUDA backends
    DeviceSettings().threading()
    DeviceSettings().cuda_backends()

    ###########################################################################################
    if model=='TimesNet':
        args.train_epochs=3
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
        args.z_loc=1
        args.z_shape=(args.pred_len + args.seq_len,args.d_model)

    # Import the trainer, main model, and student model classes dynamically
    Trainer = getattr(importlib.import_module('trainers.{}'.format(args.trainer)), 'Exp_TS2VecSupervised')
    mainModel = getattr(importlib.import_module('models.{}'.format(args.main_model['model'])), 'Model')
    studentModel = getattr(importlib.import_module('models.{}'.format(args.student_model['model'])), 'Model') \
        if args.student_model['model'] is not None else None

    ###########################################################################################

    # Load the training, validation, and test data
    train_data, train_loader = LoadData._get_data(args, 'train')
    val_data, val_loader = LoadData._get_data(args, 'val')
    test_data, test_loader = LoadData._get_data(args, 'test')

    ###########################################################################################

    # Initialize the DataLogger to log metrics, predictions, and true values
    datalogger = DataLogger(['metrics', 'preds', 'trues', 'mae', 'mse'])

    # Run the experiment for the specified number of iterations
    for ii in range(args.itr):
        metric, mae, mse, pred, true = None, None, None, None, None

        print('\n ====== Run {} ====='.format(ii))

        # Set experiment settings and seed for reproducibility
        args = PostprocessArguments().set_settings(args, exp_starttime, ii)
        DeviceSettings().seeding(args.seed + ii)

        # Initialize the trainer with the arguments, main model, and student model
        trainer = Trainer(args, mainModel, studentModel)
        print("total params for main model: ", get_total_params(trainer.main_model))
        
        if trainer.student_model is not None:
            print("total params for student model: ", get_total_params(trainer.student_model))
        
        # Start training the model
        print('\n>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
        trainer.train(train_data, train_loader, val_data, val_loader)

        # Test the model
        print('\n>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
        metric, mae, mse, pred, true, = trainer.test(test_data, test_loader)

        # Update the data logger with the results
        datalogger.update({'metrics': metric, 'preds': pred, 'trues': true, 'mae': mae, 'mse': mse})
        res=pd.DataFrame([args.y_model_main,args.seed,mae.mean(),mse.mean(),args.pred_len,args.data]).T
        res.columns=['model','seed','mae','mse','pred_len','data']
        
        output_path = "dsof-leddam.csv"
        res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
import itertools

for model,pred_len,data,seed in itertools.product(['SOFTS'],[1,24,48],['exchange','ETTh2','ETTm1','electricity','ETTh1','ETTm2','traffic','PEMS03',
                                                                                         'PEMS04','weather','PEMS07','PEMS08','solar'],[2025]):
    
        train_baseline(model,data,seed,pred_len)

            
      