import os
import torch
import argparse
import importlib
from dataloader import DatasetCustom
from torch.utils.data import DataLoader
from trainerBase import TrainerBase
from utils.tools import save_model
import random,yaml
import pandas as pd
from models.FAN_model import Model
from models.blocks.FAN import FAN
import numpy as np
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

def parse_args(model=None,data=None,data_path=None,seq_len=None,pred_len=None,seed=None):
    # 创建基础解析器
    parser = argparse.ArgumentParser()
    
    # 添加基础参数
    parser.add_argument('--seed', type=int, default=2024, help='随机数种子')
    if seed is not None:
        parser.set_defaults(seed=seed)
    parser.add_argument('--model', type=str, default='PatchTST', help='模型名称')
    if model is not None:
        parser.set_defaults(model=model)
    parser.add_argument('--data', type=str, default='ETTm1', help='数据集名称')
    if data is not None:
        parser.set_defaults(data=data)
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='数据文件名称')
    if data is not None:
        parser.set_defaults(data_path=data+'.csv')
    parser.add_argument('--features', type=str, default='M', choices=['M','S','MS'], help='特征类型')
    parser.add_argument('--631', type=bool, default=True, help='是否为631划分数据集')
    
    # 解析命令行参数以获取模型和数据名称
    cmd_args, _ = parser.parse_known_args()
    
    # 加载模型配置
    model_config_path = f'configs/model/{cmd_args.model.lower()}.yaml'
    model_args = {}
    if os.path.exists(model_config_path):
        print(f"加载模型配置: {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_args = yaml.safe_load(f)
    
    # 加载数据集配置
    data_config_path = f'configs/data/{cmd_args.data.lower()}.yaml'
    data_args = {}
    if os.path.exists(data_config_path):
        print(f"加载数据集配置: {data_config_path}")
        with open(data_config_path, 'r') as f:
            data_args = yaml.safe_load(f)
    
    # 添加其他参数定义
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    if seq_len is not None:
        parser.set_defaults(seq_len=seq_len)
    parser.add_argument('--pred_len', type=int, default=48, help='预测长度')
    if pred_len is not None:
        parser.set_defaults(pred_len=pred_len)
    parser.add_argument('--optimizer', type=str, default='adam', help='优化器')
   
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--test_run', action='store_true', default=False,
                        help='快速测试运行（仅运行少量步骤）')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='训练设备')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints2', help='模型保存路径')
    
    # 解析所有参数
    args = parser.parse_args()
    
    # 合并配置
    final_args = {}
    
    # 1. 添加模型配置
    for key, value in model_args.items():
        # 处理嵌套配置（如不同数据集的特定值）
        if isinstance(value, dict) and args.data in value:
            final_args[key] = value[args.data]
        else:
            final_args[key] = value
    
    # 2. 添加数据集配置
    for key, value in data_args.items():
        # 特殊处理特征相关的嵌套配置
        if key in ['enc_in', 'dec_in', 'c_out'] and isinstance(value, dict):
            if args.features in value:
                final_args[key] = value[args.features]
            else:
                # 使用默认值
                final_args[key] = next(iter(value.values()))
        else:
            final_args[key] = value
    
    # 3. 添加命令行参数（覆盖配置）
    for key, value in vars(args).items():
        if value is not None:
            final_args[key] = value
    
    # 确保必要参数存在
    required_params = ['output_attention', 'enc_in', 'dec_in', 'c_out', 'target']
    for param in required_params:
        if param not in final_args:
            print(f"警告: 参数 '{param}' 未定义，使用默认值")
            if param == 'output_attention':
                final_args[param] = False
            elif param in ['enc_in', 'dec_in', 'c_out']:
                final_args[param] = 1
            elif param == 'target':
                final_args[param] = 'OT'
    
    # 转换为Namespace对象
    return argparse.Namespace(**final_args)
 
import math
def main(model=None,data=None,data_path=None,seq_len=None,pred_len=None,seed=None):
    K_dict={'ETTh1':4,'ETTh2':3,
            'ETTm1':11,'ETTm2':5,'exchange':2,
            'electricity':3,'traffic':12,'weather':2}
    args = parse_args(model,data,data_path,seq_len,pred_len,seed)
    
    set_seed(args.seed)
    args.device = 'cuda:0'
    if args.data=='weather' and args.model=='iTransformer':
        args.learning_rate=0.0001
    
    # 创建模型保存路径
    os.makedirs(args.checkpoints, exist_ok=True)
    if model=='SOFTS':
        args.train_epochs=20
        args.learning_rate=0.0003
    if model=='TimesNet':
        args.train_epochs=10
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
    if model=='PEDFormer':
        args.moving_avg=7
    args.norm_style='FAN'
    
    # 加载数据集
    train_set = DatasetCustom(
        root_path=fr'dataset/{args.data}',
        data=args.data,  # 添加缺失的data参数
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, 0, args.pred_len],
        features=args.features
    )
    args.enc_in = train_set.data_x.shape[1]
    print(f"数据集特征维度: {args.enc_in}")
    val_set = DatasetCustom(
        root_path=fr'dataset/{args.data}',
        data=args.data,  # 添加缺失的data参数
        data_path=args.data_path,
        flag='val',
        size=[args.seq_len, 0, args.pred_len],
        features=args.features
    )
    
    test_set = DatasetCustom(
        root_path=fr'dataset/{args.data}',
        data=args.data,  # 添加缺失的data参数
        data_path=args.data_path,
        flag='test',
        size=[args.seq_len, 0, args.pred_len],
        features=args.features
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 初始化模型
  
    model_module = importlib.import_module(f'models.{args.model}')
    forecast_model = model_module.Model(args,args.seq_len)
    norm_model=FAN(96,args.pred_len,enc_in=args.enc_in,freq_topk=K_dict.get(args.data,5))
    model=Model(args,forecast_model,norm_model)
    
    # 初始化训练器
    trainer = TrainerBase(args, model)
    
    # 训练模型
    print(f"开始训练 {args.model} 模型...")
    print(f"数据集: {args.data}, 设备: {args.device}")
    print(f"序列长度: {args.seq_len}, 预测长度: {args.pred_len}")
    
    trained_model = trainer.train(train_loader, val_loader)
    
    # 测试模型
    print("\n测试模型性能...")
    results = trainer.test(test_loader)
    
    # 保存最终模型
    save_path = os.path.join(args.checkpoints, f'{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}_{args.norm_style}.pth')
    save_model(trained_model.state_dict(), save_path)
    
    print(f"\n训练完成! 模型已保存至: {save_path}")
    print(f"测试结果:")
    print(f"  MSE: {results[0][1]:.4f}")
    print(f"  MAE: {results[0][0]:.4f}")
    print(f"  RMSE: {results[0][2]:.4f}")
    print(f"  MAPE: {results[0][3]:.4f}%")
    print(f"  MSPE: {results[0][4]:.4f}%")
    res={
        'MSE': results[0][1],
        'MAE': results[0][0],
        'RMSE': results[0][2],
        'model':args.model,
        'data':args.data,
        'seq_len':args.seq_len,
        'pred_len':args.pred_len,
        'seed':args.seed
    }
    res=pd.DataFrame(res,index=[0])
    output_path = "experiment_itrans_FAN3.csv"
    res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
if __name__ == '__main__':
    import itertools
    for model,seed,data,pred_len in itertools.product(['iTransformer'],[2025],['traffic'],[24,48]):
        # if os.path.join('', f'{model}_{data}_{96}_{pred_len}_{seed}.pth') in os.listdir('./checkpoints2'):
            # print(f'{model}_{data}_{96}_{pred_len}_{seed}.pth'+'已存在')
        # else:
            main(model=model,seed=seed,pred_len=pred_len,data=data)