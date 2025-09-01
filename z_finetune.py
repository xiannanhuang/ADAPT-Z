import torch 
from torch import nn
from torch import optim
import argparse,yaml
import importlib
import os
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
    parser.add_argument('--batch_size', type=int, default=24, help='批量大小')
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
from tqdm import tqdm
import numpy as np
def evaluate_without_finetune(model, test_loader, device):
    """
    在不进行微调的情况下评估模型性能
    
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        preds: 所有预测值
        truths: 所有真实值
    """
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x)['pred']  # (batch_size, pred_len, channels)
            
            # 收集结果
            all_preds.append(outputs.cpu().numpy())
            all_truths.append(batch_y.cpu().numpy())
    
    # 拼接所有结果
    preds = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_truths, axis=0)
    
    return preds, truths

def calculate_metrics(preds, truths):
    """
    计算预测结果的评估指标
    
    Args:
        preds: 预测值
        truths: 真实值
        
    Returns:
        mae: 平均绝对误差
        rmse: 均方根误差
    """
    mae = np.mean(np.abs(preds - truths))
    mse = (np.mean((preds - truths) ** 2))
    return mae, mse

def extract_diagonal_band(tensor, b):
    """
    从 (n+b, n, k) 张量中提取带状区域
    
    参数:
        tensor (torch.Tensor): 输入张量，形状为 (n+b, n, k)
        b (int): 带状区域的宽度
        
    返回:
        torch.Tensor: 提取的带状区域，形状为 (b, n, k)
    """
    n = tensor.size(1)  # 获取列数
    total_rows = tensor.size(0)  # 总行数 n+b
    
    # 创建结果张量
    result = torch.zeros(b, n, tensor.size(2), device=tensor.device, dtype=tensor.dtype)
    
    # 使用循环提取每列的值
    for col in range(n):
        # 计算起始行: 第0列从第(n-1)行开始，依次递减
        start_row = (n - 1) - col
        
        # 对于最后一列，直接从第0行开始取
        if start_row < 0:
            start_row = 0
            
        # 提取从start_row开始的b行数据
        band_data = tensor[start_row:start_row+b, col, :]
        
        # # 处理边缘情况：当提取的数据不足b行时，用零填充
        # if band_data.size(0) < b:
        #     padding = torch.zeros(b - band_data.size(0), tensor.size(2), 
        #                         device=tensor.device, dtype=tensor.dtype)
        #     band_data = torch.cat([band_data, padding], dim=0)
        
        # 将提取的数据放入结果张量的相应位置
        result[:, col, :] = band_data
    
    return result
batch_size_dict={
    'ETTh1':24,
    'ETTh2':24,
    'ETTm1':24*4,
    'ETTm2':24*4,
    'PEMS03':12,
    'PEMS04':12,
    'PEMS07':12,
    'PEMS08':12,
    'traffic':24,
    'solar':6*24,
    'exchange':7,
    'weather':24,
    'electricity':24
}
class online_finetune_z():
    def __init__(self, model, args,z_shape,z_loc,ogd_step=1,ema=0.7):
        self.model = model
        self.args = args
        self.z=nn.Parameter(torch.zeros(z_shape,requires_grad=True,device=args.device))
        
        self.optimizer = optim.Adam([self.z], lr=0.001)
        self.loss_function = nn.MSELoss()
        self.ogd_step = ogd_step
        self.ema=ema
        self.z_loc=z_loc
    
        
    def online_finetune(self, test_loader):
        if self.optimizer is None:
            print("没有需要更新的参数，跳过在线微调")
            return None, None
            
        preds = []
        truths = []
   
    
        batch_x_list=[]
        batch_y_list=[]
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.to(self.args.device)
            batch_y = batch_y.to(self.args.device)
            batch_x_mark = batch_x_mark.to(self.args.device)
            batch_y_mark = batch_y_mark.to(self.args.device)
            outputs = self.model(batch_x,z=self.z,z_loc=self.z_loc)['pred']  #(batch_size, pred_len, channels)
            batch_x_list.append(batch_x)
            batch_y_list.append(batch_y)
            batch_x_list=batch_x_list[-100:]
            batch_y_list=batch_y_list[-100:]
            preds.append(outputs.detach().cpu().numpy())
            truths.append(batch_y.detach().cpu().numpy())
            if torch.cat(batch_x_list[-100:], dim=0).shape[0] > self.args.batch_size+self.args.pred_len:
                z_old=self.z.clone()
                for _ in range(self.ogd_step):
                    batch_x = torch.cat(batch_x_list[-100:], dim=0)[-self.args.batch_size:]
                    batch_y = torch.cat(batch_y_list, dim=0)[-self.args.batch_size:]
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x,z=self.z,z_loc=self.z_loc)['pred']  #(batch_size, pred_len, channels)
                    loss = self.loss_function(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                self.z.data = self.z.data * (1 - self.ema) + z_old * (self.ema)
                
                
            
            
                
                
            
            
        
        preds = np.concatenate(preds, axis=0)
        truths = np.concatenate(truths, axis=0)
       
        
        return preds, truths
import math
# 修改模块路径访问方式
def get_module_by_path(model, path):
    """按点分隔的路径安全地获取模块"""
    module = model
    parts = path.split('.')
    
    for part in parts:
        # 处理数字索引（如列表中的模块）
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module
import pandas as pd
def finetune(model=None,data=None,data_path=None,seq_len=None,pred_len=None,seed=None,ogd_step=0,ema=0.7):
    from torch.utils.data import DataLoader
    import os
    from dataloader import DatasetCustom
    # 解析参数（假设已定义）
    args = parse_args(model,data,data_path,seq_len,pred_len,seed)
    args.device = 'cuda:0'
    
    # 创建模型保存路径
    os.makedirs(args.checkpoints, exist_ok=True)
    
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
    if model=='TimesNet':
        args.train_epochs=10
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
        args.z_loc=1
        args.z_shape=(args.pred_len + args.seq_len,args.d_model)
    args.batch_size = batch_size_dict[args.data]
    args.batch_size = 1
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # 加载模型
    model_path = fr'checkpoints2\{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}.pth'
    
    try:
        model_module = importlib.import_module(f'models.{args.model}'+'_z')
        model = model_module.Model(args, args.seq_len)
    except (ImportError, AttributeError) as e:
        print(f"错误: 无法加载模型 {args.model}")
        print(f"详情: {e}")
        exit()
    
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    print(f'成功加载模型: {model_path}')
    
    # 获取所有参数名称
    param_names = [name for name, _ in model.named_parameters()]
    target_dict={'iTransformer':[1],'SOFTS':[2]
        }
    # target_shape={
    #     'iTransformer':{'input':(args.batch_size,args.seq_len,args.enc_in), 'output':(args.batch_size,args.enc_in, args.pred_len),
    #                     1:(args.batch_size,args.enc_in,args.d_model),2:(args.batch_size,args.enc_in,args.d_model)
    #                     ,3:(args.batch_size,args.enc_in,args.d_model),'emb':(args.batch_size,args.enc_in,args.d_model)},
         
    # }
    target_shape={
        'iTransformer':{'input':(args.seq_len,args.enc_in), 'projection':( args.pred_len,args.enc_in),
                        1:(args.enc_in,args.d_model),2:(args.enc_in,args.d_model)
                        ,3:(args.enc_in,args.d_model),'emb':(args.enc_in,args.d_model)},
        'SOFTS':{1:(args.enc_in,256),2:(args.enc_in,256),3:(args.enc_in,256),'emb':(args.enc_in,256)},
        # 'TimesNet':{1:(args.pred_len + args.seq_len,args.d_model)},
         
    }
    metrics={}
    if args.model=='iTransformer':
        target_layers = [args.e_layers-1]
    if args.model=='SOFTS':
        target_layers = [2]
    if args.model=='TimesNet':
        target_layers = [1]
    for target_layer in target_layers:
        # if isinstance(target_layer, int):
        #     if args.e_layers<target_layer:
        #         continue
        if target_layer=='input':
            finetuner=online_finetune_z(model,args,target_shape[args.model][target_layer],ogd_step=ogd_step,z_loc=target_layer,ema=ema)
        else:
            finetuner=online_finetune_z(model,args,target_shape[args.model][target_layer],ogd_step=ogd_step,z_loc=target_layer,ema=ema)
        preds, truths = finetuner.online_finetune(test_loader)
        mae, mse = calculate_metrics(preds, truths)
        print(f"层 {target_layer} 微调结果 - MAE: {mae:.4f}, MSE: {mse:.4f}\n")
        metrics[target_layer] = (mae, mse)

    
    res=pd.DataFrame(metrics).T
    res.columns=['mae','mse']
    res['model']=[args.model]*len(res)
    res['seed']=[args.seed]*len(res)
    res['pred_len']=[args.pred_len]*len(res)
    res['data']=[args.data]*len(res)
    res['ogd_step']=[ogd_step]*len(res)
    res['ema']=[ema]*len(res)
    res['layer']=res.index

    output_path = "softs_fogd_sgd.csv"
    res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    
if __name__ == "__main__":
    import itertools
    import pandas as pd
    # df=pd.read_csv('z_experiment_results_itrans.csv')
    # done_experiments = df[['model','seed','pred_len','data','ogd_step','ema']].drop_duplicates()
    # done_experiments['done']=done_experiments.apply(lambda x: ' '.join(str(x) for x in x.values),axis=1)
    # done_experiments=done_experiments['done'].to_list()
    for model,seed,data,pred_len,ogd_step,ema in itertools.product(['iTransformer'],[2025],['exchange','ETTh2','ETTm1','electricity','ETTh1','ETTm2','traffic','PEMS03',
                                                                                         'PEMS04','weather','PEMS07','PEMS08','solar'], [1,24,48],[1],[0.]):
            finetune(model=model,seed=seed,pred_len=pred_len,data=data,ogd_step=ogd_step,ema=ema)
    # else:
    #         print(f'实验 {model} {seed} {pred_len} {data} {ogd_step} {ema} 已完成')

 