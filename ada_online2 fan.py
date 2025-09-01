import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import argparse
import importlib
import random
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from models.FAN_model_z import Model
from models.blocks.FAN import FAN
settings={
    'iTransformer':
        {
            'ETTh1':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
                 },
        'ETTh2':{
             'online_lr':0.000003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
               },
         'ETTm1':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
              },
        'ETTm2':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
           'adapter_lr':0.001,
            'finetune_lr':0.00003
               },
         'PEMS03':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
         'adapter_lr':0.001,
            'finetune_lr':0.00003
                 },
        'PEMS04':{
             'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
        },
          'PEMS07':{
             'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
                 },
        'PEMS08':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
               },
         'traffic':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
              },
        'weather':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
               },
         'solar':{
             'online_lr':0.00003,
            'adp_online_lr':0.0003,
            'adapter_lr':0.001,
            'finetune_lr':0.00003
                 },
        'electricity':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
          'adapter_lr':0.001,
            'finetune_lr':0.00003
        },
        'exchange':{
            'online_lr':0.00003,
            'adp_online_lr':0.0003,
           'adapter_lr':0.001,
            'finetune_lr':0.00003
        },
         },
    'SOFTS':{
        
            'ETTh1':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'ETTh2':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'ETTm1':{
           'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
              },
        'ETTm2':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'PEMS03':{
           'online_lr':0.00003,
            'adapter_lr':0.003,
            'adp_online_lr':0.001,
                 },
        'PEMS04':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
        },
          'PEMS07':{
             'online_lr':0.00003,
            'adapter_lr':0.003,
            'adp_online_lr':0.001,
                 },
        'PEMS08':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'traffic':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
              },
        'weather':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'solar':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'electricity':{
          'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
        },
        'exchange':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
        },
         },
    'TimesNet':{
        
            'ETTh1':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'ETTh2':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'ETTm1':{
           'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
              },
        'ETTm2':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'PEMS03':{
           'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'PEMS04':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
        },
          'PEMS07':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'PEMS08':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'traffic':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
              },
        'weather':{
            'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
               },
         'solar':{
             'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
                 },
        'electricity':{
          'online_lr':0.00003,
            'adapter_lr':0.001,
            'adp_online_lr':0.0003,
        },
        'exchange':{
            'online_lr':0.0000,
            'adapter_lr':0.0003,
            'adp_online_lr':0.0001,
        },
         }
    
}
class Ada_fineture:
    def __init__(self, model, d_model,args):
        self.model=model
        self.device=args.device
        self.adapter=self.bulid_adapter(d_model,args)
        self.args=args
        self.optimizer=optim.Adam(self.adapter.parameters(),lr=0.0003)
        self.loss_func=nn.MSELoss()
        self.device=args.device
        self.adapter.to(self.device)
        self.z=nn.Parameter(torch.zeros(args.z_shape,requires_grad=True,device=args.device))

    def bulid_adapter(self,d_model,args):
        class Adapter(nn.Module):
            def __init__(self,d_model,args):
                super(Adapter,self).__init__()
                self.args=args
                self.linear=nn.Linear(self.args.z_shape[-1],64)
                self.linear_=nn.Linear(d_model,64)
                self.linear2=nn.Linear(64,64)
                self.linear3=nn.Linear(64,self.args.z_shape[-1])
                # lr=np.ones((args.enc_in,d_model))
                self.lr=nn.Parameter(torch.ones(self.args.z_shape,dtype=torch.float32),requires_grad=True)
            def forward(self,x,his_grad):
                his_grad=(his_grad*self.lr).unsqueeze(0).repeat(x.size(0),1,1)
             
                his_grad=self.linear_(his_grad)
                # x=torch.cat([x,his_grad],dim=-1)
                # x=his_grad
                # x=x+his_grad
                x=self.linear(x)
                # x=his_grad
                x=x+his_grad
                x=torch.relu(x)
                x=self.linear2(x)
                x=torch.relu(x)
                x=self.linear3(x)
                # return -his_grad*0.5
              
                return x
        return Adapter(d_model,args)
    
    def load(self):
       
        path1=fr'checkpoints2/{self.args.model}_{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.seed}_with_tr_a.pth'
        self.adapter.load_state_dict(torch.load(path1))
        path2=fr'checkpoints2/{self.args.model}_{self.args.data}_{self.args.seq_len}_{self.args.pred_len}_{self.args.seed}_with_tr_m.pth'
        self.model.load_state_dict(torch.load(path2))
    
    def train(self, train_loader, val_loader):
        print("start train")
        best_mse=100
        flag=0
        for _ in range(3):
            adapter_state=self.adapter.state_dict()
            i=int((random.random()*train_loader.batch_size))
            # 创建子集：从索引 i 开始到数据集末尾
            subset = Subset(train_loader.dataset, indices=range(i, len(train_loader.dataset)))

            # 创建新的 DataLoader
            train_loader2 = DataLoader(
                subset,
                batch_size=train_loader.batch_size,  # 保持与原 loader 相同的 batch_size
                shuffle=False,                       # 确保不洗牌以保持顺序
                num_workers=train_loader.num_workers,  # 可选：保持 worker 数量
                pin_memory=train_loader.pin_memory     # 可选：保持内存锁页设置
            )
            mse=self.val(train_loader2,mode='train')
            model_state=self.model.state_dict()
            mse=self.val(val_loader)
            print(mse)
            self.model.load_state_dict(model_state)
            if mse<best_mse:
                best_mse=mse.copy()
                best_adapter_state=adapter_state.copy()
                best_model_state=model_state.copy()
            else:
                flag+=1
                print(f'loss not decent for {flag} times')
                if flag==3:
                    break
        self.adapter.load_state_dict(best_adapter_state)
        self.model.load_state_dict(best_model_state)



    def forward(self,x,history_grad):
        z=torch.zeros(self.args.z_shape).to(self.device)
        output=self.model(x,z_loc=self.args.z_loc)
        feature=output['feature']

        z=self.adapter(feature,history_grad)
        output=self.model(x,z=z,z_loc=self.args.z_loc)
        pred=output['pred']
        return pred

    def val(self, val_loader,mode='val'):
        print("start val")
        his_grad=torch.zeros(self.args.z_shape).to(self.device)
        preds=[]
        truths=[]
        x_list=[]
        if mode=='val':
            self.optimizer2=optim.Adam(self.model.parameters(),lr=settings[self.args.model][self.args.data]['online_lr'])
            self.optimizer=optim.Adam(self.adapter.parameters(),lr=settings[self.args.model][self.args.data]['adapter_lr'])
        if mode=='train':
            self.optimizer2=optim.Adam(self.model.parameters(),lr=settings[self.args.model][self.args.data]['finetune_lr'])
            self.optimizer=optim.Adam(self.adapter.parameters(),lr=settings[self.args.model][self.args.data]['adapter_lr'])
        
        for i, (x, y,_,_) in tqdm(enumerate(val_loader),total=len(val_loader)):
            x=x.to(self.device)
            y=y.to(self.device)
            x_list.append(x.detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy())
            output=self.model(x,z_loc=self.args.z_loc,z=self.z)
            feature=output['feature']
            z=self.adapter(feature,his_grad)
            output=self.model(x,z=z,z_loc=self.args.z_loc)
            pred=output['pred']
            preds.append(pred.detach().cpu().numpy())
            loss=self.loss_func(pred,y)
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer.step()
            if mode=='train':
                self.optimizer2.step()
            
            if len(np.concatenate(preds,axis=0))>self.args.pred_len+self.args.batch_size-1:
                if self.args.pred_len!=1:
                    x_batch=np.concatenate(x_list[-100:],axis=0)[(-self.args.pred_len-self.args.batch_size+1):(-self.args.pred_len+1)]
                    y_batch=np.concatenate(truths[-100:],axis=0)[-self.args.pred_len-self.args.batch_size+1:-self.args.pred_len+1]
                else:
                    x_batch=np.concatenate(x_list[-100:],axis=0)[(-self.args.pred_len-self.args.batch_size+1):]
                    y_batch=np.concatenate(truths[-100:],axis=0)[-self.args.pred_len-self.args.batch_size+1:]
                x_batch=torch.from_numpy(x_batch).to(self.device)
                y_batch=torch.from_numpy(y_batch).to(self.device)
                output=self.model(x_batch,z_loc=self.args.z_loc,z=self.z)
                feature=output['feature']
                pred_batch=output['pred']
                feature.retain_grad() 

               
                loss=self.loss_func(pred_batch,y_batch)
                
                loss.backward()
                his_grad2=self.z.grad
                his_grad=torch.mean(his_grad2,dim=0)

        print("end val")
        mae,  mse = calculate_metrics(np.concatenate(preds,axis=0), np.concatenate(truths,axis=0))
        print("mae: ",mae)
        print("mse: ",mse)
        return mse
    
    def online(self, test_loader):
        pred=[]
        truth=[]
        x_list=[]
        self.optimizer2=optim.Adam(self.model.fm.projector.parameters(),lr=settings[self.args.model][self.args.data]['online_lr'])
        self.optimizer=optim.Adam(self.adapter.parameters(),lr=settings[self.args.model][self.args.data]['adp_online_lr'])
        # 统计 optimizer 的参数数量
        optimizer_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)

        # 统计 optimizer2 的参数数量
        optimizer2_params = sum(p.numel() for p in self.model.fm.projector.parameters() if p.requires_grad)

        # 打印总数
        total_params = optimizer_params + optimizer2_params
        print(f"Total trainable parameters in both optimizers: {total_params:,}")
        print("start online")
        his_grad=torch.zeros(self.args.z_shape).to(self.device)
        his_grads=[]
        features=[]
        his_grad_dict={}
        feature_dict={}
        x_dict={}
        y_dict={}
        for i, (x, y,_,_) in tqdm(enumerate(test_loader),total=len(test_loader)):
            x=x.to(self.device)
            y=y.to(self.device)
            pred_i=self.model(x,z_loc=self.args.z_loc,z=torch.zeros(self.args.z_shape).to(self.device))
            feature=pred_i['feature']
            z=self.adapter(feature,his_grad)
            output=self.model(x,z=z,z_loc=self.args.z_loc)
            # output=self.model(x)
            pred_i=output['pred']
            pred.append(pred_i.detach().cpu().numpy())
            truth.append(y.detach().cpu().numpy())
            x_list.append(x.detach().cpu().numpy())
            his_grads.append(his_grad.detach().cpu().numpy())
            features.append(feature.detach().cpu().numpy())
            his_grad_dict[(i*test_loader.batch_size)]=his_grad.detach().cpu().numpy()
            feature_dict[(i*test_loader.batch_size)]=feature.detach().cpu().numpy()
            x_dict[(i*test_loader.batch_size)]=x.detach().cpu().numpy()
            y_dict[(i*test_loader.batch_size)]=y.detach().cpu().numpy()
            if i*test_loader.batch_size>300:
                del his_grad_dict[(i*test_loader.batch_size-(300//test_loader.batch_size)*test_loader.batch_size)]
                del feature_dict[(i*test_loader.batch_size-(300//test_loader.batch_size)*test_loader.batch_size)]
                del x_dict[(i*test_loader.batch_size-(300//test_loader.batch_size)*test_loader.batch_size)]
                del y_dict[(i*test_loader.batch_size-(300//test_loader.batch_size)*test_loader.batch_size)]
            if len(np.concatenate(pred,axis=0))>self.args.pred_len+self.args.batch_size:
                if self.args.pred_len!=1:
                    x_batch=np.concatenate(x_list[-100:],axis=0)[(-self.args.pred_len-self.args.batch_size+1):(-self.args.pred_len+1)]
                    y_batch=np.concatenate(truth[-100:],axis=0)[-self.args.pred_len-self.args.batch_size+1:-self.args.pred_len+1]
                else:
                    x_batch=np.concatenate(x_list[-100:],axis=0)[(-self.args.pred_len-self.args.batch_size+1):]
                    y_batch=np.concatenate(truth[-100:],axis=0)[-self.args.pred_len-self.args.batch_size+1:]
                x_batch_index=i*test_loader.batch_size-self.args.pred_len-self.args.batch_size+1
                x_batch=torch.from_numpy(x_batch).to(self.device)
                y_batch=torch.from_numpy(y_batch).to(self.device)
                output=self.model(x_batch,z_loc=self.args.z_loc,z=self.z)
                feature=output['feature']
                pred_batch=output['pred']
                feature.retain_grad() 

               
                loss=self.loss_func(pred_batch,y_batch)
                loss.backward(retain_graph=True)
                his_grad2=self.z.grad
                his_grad=torch.mean(his_grad2,dim=0)
                if x_batch_index>=0:
                    for _ in range(1):
                        index=i*test_loader.batch_size-test_loader.batch_size*((self.args.pred_len-test_loader.batch_size)//(test_loader.batch_size)+1)
                        x_batch=x_dict[index]
                        y_batch=y_dict[index]
                        feature=feature_dict[index]
                        his_grad2=his_grad_dict[index]
                        x_batch=torch.from_numpy(x_batch).to(self.device)
                        y_batch=torch.from_numpy(y_batch).to(self.device)
                        feature=torch.from_numpy(feature).to(self.device)
                        his_grad2=torch.from_numpy(his_grad2).to(self.device)
                        z=self.adapter(feature,his_grad2)
                        output=self.model(x_batch,z=z,z_loc=self.args.z_loc)
                        pred_batch=output['pred']
                        loss=self.loss_func(pred_batch,y_batch)
                        self.optimizer.zero_grad()
                        self.optimizer2.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer2.step()


                


        pred=np.concatenate(pred,axis=0)
        truth=np.concatenate(truth,axis=0)
        mae=np.mean(np.abs(pred-truth))
        mse=np.mean((pred-truth)**2)
            
        print("end online")
        return mae,mse
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
def set_seed(seed: int):
    """
    设置随机数种子以获得可重复的结果
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # random.seed(seed)
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
    'solar':24,
    'exchange':7,
    'weather':24,
    'electricity':24
}
import pandas as pd
import math
def finetune(model=None,data=None,data_path=None,seq_len=None,pred_len=None,seed=None,ogd_step=0,ema=0.7,z_loc=1):
    from torch.utils.data import DataLoader
    import os
    from dataloader import DatasetCustom
    # 解析参数（假设已定义）
    args = parse_args(model,data,data_path,seq_len,pred_len,seed)
    args.device = 'cuda:0'
    K_dict={'ETTh1':4,'ETTh2':3,
            'ETTm1':11,'ETTm2':5,'exchange':2,
            'electricity':3,'traffic':12,'weather':2}
    
    # 创建模型保存路径
    os.makedirs(args.checkpoints, exist_ok=True)
    target_shape={
        'iTransformer':{'input':(args.batch_size,args.seq_len,args.enc_in), 'output':(args.batch_size,args.enc_in, args.pred_len),
                        1:(args.batch_size,args.enc_in,args.d_model),2:(args.batch_size,args.enc_in,args.d_model)
                        ,3:(args.batch_size,args.enc_in,args.d_model),'emb':(args.batch_size,args.enc_in,args.d_model)},
         
    }
    
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
    # args.batch_size = batch_size_dict[args.data]
    args.norm_style='FAN'
    set_seed(args.seed)
    if model=='TimesNet':
        args.train_epochs=10
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
        args.z_loc=1
        args.z_shape=(args.pred_len + args.seq_len,args.d_model)


    
   
    train_loader = DataLoader(train_set, batch_size=24, shuffle=False)
    args.batch_size = 24
    val_loader = DataLoader(val_set, batch_size=24, shuffle=False)
    
    test_loader = DataLoader(test_set, batch_size=24, shuffle=False)
    if args.model=='SOFTS':
        args.z_loc=z_loc
        args.z_shape=(args.enc_in,256)
        # 加载模型
    if args.model=='iTransformer':
        args.z_loc=args.e_layers-1
        args.z_shape=(args.enc_in,args.d_model)
    # 加载模型
    model_path = fr'checkpoints2\{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}_{args.norm_style}.pth'
    
    try:
        model_module = importlib.import_module(f'models.{args.model}'+'_z')
        forecast_model = model_module.Model(args,args.seq_len)
        norm_model=FAN(96,args.pred_len,enc_in=args.enc_in,freq_topk=K_dict.get(args.data,5))
        model=Model(args,forecast_model,norm_model)
        
    except (ImportError, AttributeError) as e:
        print(f"错误: 无法加载模型 {args.model}")
        print(f"详情: {e}")
        exit()
    
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    print(f'成功加载模型: {model_path}')
    if args.model=='SOFTS':
        adapter=Ada_fineture(model, 256,args)
    else:
        adapter=Ada_fineture(model, args.d_model,args)
    # adapter.train(train_loader,val_loader)
    # for _ in range(3):
    #     adapter.val(val_loader)
    # torch.save(adapter.model.state_dict(), fr'checkpoints2\{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}_with_tr_m.pth')
    # torch.save(adapter.adapter.state_dict(), fr'checkpoints2\{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.seed}_with_tr_a.pth')
  
    # adapter.load()
    
    for _ in range(3):
        adapter.val(val_loader)
    mae,mse=adapter.online(test_loader)
    print(f'测试集mae:{mae},mse:{mse}')
    res={'model':args.model,'data':args.data,'seq_len':args.seq_len,'pred_len':args.pred_len,'seed':args.seed,'mae':mae,'mse':mse,'z_loc':z_loc}
    res=pd.DataFrame(res,index=[0])
    output_path = "itrans_adap_FAN.csv"
    res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    
    return mae,mse 
import itertools
for model,data,pred_len,seed,z_loc in itertools.product(['iTransformer'],['exchange','ETTh2','ETTm1','electricity','ETTh1','ETTm2','traffic','PEMS03',
                                                                                         'PEMS04','weather','PEMS07','PEMS08','solar'],[1,24,48],[2025],[2]):
    # df=pd.read_csv('softs_adapter2.csv')
    # done_experiments = df[['pred_len','data','seq_len','seed','z_loc']].drop_duplicates()
    # done_experiments['done']=done_experiments.apply(lambda x: '_'.join(str(x) for x in x.values),axis=1)
    # done_experiments=done_experiments['done'].to_list()
    # done_experiments=[i for i in done_experiments]
    # if f'{pred_len}_{data}_{96}_{seed}_{z_loc}' in done_experiments:
    #         print(f'{pred_len}_{data}_{96}_{seed}_{z_loc} already done')
    # else:
        finetune(model,data=data,pred_len=pred_len,seed=seed,z_loc=z_loc)