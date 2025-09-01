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
class online_finetune():
    def __init__(self, model, args, parameter_name=None,ogd_step=1):
        self.model = model
        self.args = args
        self.parameter_name = parameter_name  # 保存参数名称
        
        # 冻结所有参数
        # for p in self.model.parameters():
        #     p.requires_grad = False
            
        # 只解冻目标参数
        if parameter_name is not None:
            self._unfreeze_target_param()
        # self.args.optimizer = 'sgd'
        self.optimizer = self._select_optimizer()
        self.loss_function = nn.MSELoss()
        self.ogd_step = ogd_step
        
    def _unfreeze_target_param(self):
        """解冻目标参数层"""
        if self.parameter_name=='all':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                print(f"解冻参数层: {name}")
            return
        for name, param in self.model.named_parameters():
            if name == self.parameter_name:
                param.requires_grad = True
                print(f"解冻参数层: {name}")
    
    def _select_optimizer(self):
        # 获取需要更新的参数列表
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        
        if not params_to_update:
            return None
        
       
        optimizer = optim.SGD(params_to_update, lr=0.000003)
        return optimizer
        
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
            outputs = self.model(batch_x) #(batch_size, pred_len, channels)
            if isinstance(outputs, dict):
                outputs = outputs['pred']
            batch_x_list.append(batch_x)
            batch_y_list.append(batch_y)
            preds.append(outputs.detach().cpu().numpy())
            truths.append(batch_y.detach().cpu().numpy())
            batch_x_list=batch_x_list[-100:]
            batch_y_list=batch_y_list[-100:]
            if torch.cat(batch_x_list, dim=0).shape[0] > self.args.batch_size+self.args.pred_len:
                for _ in range(self.ogd_step):
                    batch_x = torch.cat(batch_x_list, dim=0)[-self.args.batch_size-self.args.pred_len:-self.args.pred_len]
                    batch_y = torch.cat(batch_y_list, dim=0)[-self.args.batch_size-self.args.pred_len:-self.args.pred_len]
                    
                    self.optimizer.zero_grad()

                    outputs = self.model(batch_x) #(batch_size, pred_len, channels)
                    if isinstance (outputs, dict):
                        outputs = outputs['pred']
                    loss = self.loss_function(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                
            
            
                
                
            
            
        
        preds = np.concatenate(preds, axis=0)
        truths = np.concatenate(truths, axis=0)
        # preds=test_loader.dataset.scale.inverse_transform(preds)
        # truths=test_loader.scale.inverse_transform(truths)
        
        mae = np.mean(np.abs(preds - truths))
        mse = (np.mean((preds - truths) ** 2))
        print(f'在线微调结果 - 参数: {self.parameter_name if self.parameter_name else "所有参数"} | MAE: {mae:.4f}, RMSE: {mse:.4f}')
        
        return preds, truths
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
import math
def finetune(model=None,data=None,data_path=None,seq_len=None,pred_len=None,seed=None,ogd_step=1,lr=0.001):
    from torch.utils.data import DataLoader
    import os
    from dataloader import DatasetCustom
    # 解析参数（假设已定义）
    args = parse_args(model,data,data_path,seq_len,pred_len,seed)
    args.device = 'cuda:0'
    args.learning_rate = lr
    if model=='TimesNet':
        args.train_epochs=10
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
    
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
    
    # args.batch_size = 1
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # 加载模型
    model_path = fr'checkpoints2\{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}.pth'
    
    try:
        model_module = importlib.import_module(f'models.{args.model}')
        model = model_module.Model(args, args.seq_len)
    except (ImportError, AttributeError) as e:
        print(f"错误: 无法加载模型 {args.model}")
        print(f"详情: {e}")
        exit()
    
    model.load_state_dict(torch.load(model_path))
    print(f'成功加载模型: {model_path}')
  
    # 获取所有参数名称
    param_names = [name for name, _ in model.named_parameters()]
    print(f"模型包含 {len(param_names)} 个可训练参数层:")
    target_dict={'iTransformer':[
        
        ["encoder.attn_layers.0.conv2", "encoder.attn_layers.0.conv1","encoder.attn_layers.0.attention"],
        
        ["encoder.attn_layers.1.conv2","encoder.attn_layers.1.conv1","encoder.attn_layers.1.attention"],
        
        ["encoder.attn_layers.2.conv2","encoder.attn_layers.2.conv1","encoder.attn_layers.2.attention"],
        "enc_embedding.value_embedding",
        'projector'],
        'DLinear':['Linear_Seasonal','Linear_Trend'],
        'PatchTST':['model.backbone.encoder.layers.0.self_attn','model.backbone.encoder.layers.1.self_attn','model.backbone.encoder.layers.2.self_attn',
                    'model.backbone.encoder.layers.0.ff','model.backbone.encoder.layers.1.ff','model.backbone.encoder.layers.2.ff']}
    # target_layers = target_dict[args.model]
    target_layers = ['projectio']
    # 筛选出这些层的所有参数
    print("\n选择进行微调的层:")
    for i, name in enumerate(target_layers, 1):
        print(f"{i}. {name}")

    # 创建实验结果存储
    results = {}
    metrics = {}  # 同时存储指标结果

  
    # 创建原始模型的副本
    baseline_model = model_module.Model(args, args.seq_len)
    baseline_model.load_state_dict(torch.load(model_path))
    baseline_model.to(args.device)

   

    # 2. 对每个目标层单独进行OGD实验（按层微调）
    for i,layer_spec in enumerate(target_layers):
        # 创建新的模型副本
        model_copy = model_module.Model(args, args.seq_len)
        model_copy.load_state_dict(torch.load(model_path))
        model_copy.to(args.device)
        
        print(f"\n{'='*50}")
        print(f"开始微调层: {layer_spec}")
        print(f"{'='*50}\n")
        
        # 冻结所有参数
        for param in model_copy.parameters():
            param.requires_grad = False
        
        # 仅解冻目标层的参数
        layer = model_copy
        # 确定层名称（用于日志）
        if isinstance(layer_spec, list):
            layer_name = f"组合层_{i+1}"
            layer_paths = layer_spec
        else:
            layer_name = layer_spec
            layer_paths = [layer_spec]
        
        # print(f"\n{'='*50}")
        # print(f"开始微调: {layer_name}")
        # print(f"{'='*50}\n")
        
        # 解冻目标层
        total_trainable = 0
        for path in layer_paths:
            try:
                # 安全获取目标层模块
                target_module = get_module_by_path(model_copy, path)
                
                # 解冻该层的所有参数
                for param in target_module.parameters():
                    param.requires_grad = True
                
                # 统计参数量
                trainable_params = sum(p.numel() for p in target_module.parameters())
                total_trainable += trainable_params
                print(f"解冻层 {path} ({trainable_params} 参数)")
            
            except Exception as e:
                print(f"错误: 无法访问层 {path}. 原因: {str(e)}")
                continue
    
        
        # 统计解冻的参数量
        trainable_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_copy.parameters())
        print(f"解冻层 {layer_name} 的所有参数 ({trainable_params}/{total_params} 参数可训练)")
        
        # 创建微调器
        finetuner = online_finetune(model_copy, args,ogd_step=ogd_step)
        
        # 执行在线微调
        preds, truths = finetuner.online_finetune(test_loader)
        
        # 保存结果
        if preds is not None and truths is not None:
            results[layer_name] = (preds, truths)
            
            # 计算并保存指标
            mae, mse = calculate_metrics(preds, truths)
            metrics[layer_name] = (mae, mse)
            print(f"层 {layer_name} 微调结果 - MAE: {mae:.4f}, MSE: {mse:.4f}\n")

    # 3. 完整模型的微调
    print(f"\n{'='*50}")
    print("开始完整模型的在线微调")
    print(f"{'='*50}\n")

    model_full = model_module.Model(args, args.seq_len)
    model_full.load_state_dict(torch.load(model_path))
    model_full.to(args.device)

    finetuner_full = online_finetune(model_full, args, 'all',ogd_step=ogd_step)
    full_preds, full_truths = finetuner_full.online_finetune(test_loader)

    # 保存完整模型微调结果
    if full_preds is not None and full_truths is not None:
        results["Full Model Finetune"] = (full_preds, full_truths)
        
        # 计算并保存指标
        full_mae, full_mse = calculate_metrics(full_preds, full_truths)
        metrics["Full Model Finetune"] = (full_mae, full_mse)
        print(f"完整模型微调结果 - MAE: {full_mae:.4f}, MSE: {full_mse:.4f}\n")

    # 4. 比较所有实验结果
    print("\n实验结果比较:")
    print(f"{'微调方法':<50} | {'MAE':<10} | {'MSE':<10} | {'与基线MAE差':<15} | {'与基线MSE差':<15}")
    print("-" * 120)

  
    res=pd.DataFrame(metrics).T
    res.columns=['MAE','MSE']
    res['tune_parameters']=res.index
    res['model']=[args.model]*len(res)
    res['pred_len']=[args.pred_len]*len(res)
    res['seed']=[args.seed]*len(res)
    res['data']=[args.data]*len(res)
    res['ogd_step']=[ogd_step]*len(res)

    output_path = "OGD_trans_fina.csv"
    res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
if __name__ == "__main__":
    import itertools
    import pandas as pd
    # if os.exiting('OGD_SOFTS.csv'):
    # df=pd.read_csv('OGD_times.csv')
    # else:
    #     df=pd.DataFrame()
    # done_experiments = df[['model','seed','pred_len','data','ogd_step']].drop_duplicates()
    # done_experiments['done']=done_experiments.apply(lambda x: ' '.join(str(x) for x in x.values),axis=1)
    # done_experiments=done_experiments['done'].to_list()
    for model,seed,pred_len,data,ogd_step in itertools.product(['iTransformer'],[2025],[1,24,48],['solar','exchange','ETTh1','ETTh2','ETTm1','ETTm2','PEMS03',
                                               'PEMS04','PEMS07','PEMS08','traffic','weather','electricity'],[1,24,48]):
       
            finetune(model=model,seed=seed,pred_len=pred_len,data=data,ogd_step=ogd_step,lr=0.001)
       

 