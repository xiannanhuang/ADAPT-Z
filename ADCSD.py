import numpy as np
import torch
from scipy.fft import fft, fftfreq
from sklearn.metrics import mean_squared_error, mean_absolute_error

from z_finetune import parse_args
import yaml
import os
import yaml
import torch
import numpy as np
import argparse
from sklearn.linear_model import Ridge
from collections import defaultdict, deque
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import importlib
import time
import numpy as np
import torch
import os

from ray import tune

import scipy.sparse as sp

from tqdm import tqdm


class moving_avg(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.avg = torch.nn.AvgPool2d(kernel_size=(1, kernel_size), stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ADCSDModule(torch.nn.Module):
    def __init__(self, output_dim, output_window, num_nodes, moving_avg=5):
        super(ADCSDModule, self).__init__()
        self.decomp = series_decomp(moving_avg)
        hidden_ratio = 128
        FWL_list_1 = [torch.nn.Linear(output_dim, hidden_ratio), 
                      torch.nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
                      torch.nn.GELU(), 
                      torch.nn.Linear(hidden_ratio, output_dim)]
        self.FWL_1 = torch.nn.Sequential(*FWL_list_1)
        self.learned_lambda_1 = torch.nn.Parameter(torch.zeros(num_nodes, 1))
        FWL_list_2 = [torch.nn.Linear(output_dim, hidden_ratio), 
                      torch.nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
                      torch.nn.GELU(), 
                      torch.nn.Linear(hidden_ratio, output_dim)]
        self.FWL_2 = torch.nn.Sequential(*FWL_list_2)
        self.learned_lambda_2 = torch.nn.Parameter(torch.zeros(num_nodes, 1))

    def forward(self, x):
        output_1, output_2 = self.decomp(x)
        output = x + self.learned_lambda_1 * self.FWL_1(output_1) + self.learned_lambda_2 * self.FWL_2(output_2)
        return output[:,:,:,:]




   
    


def ADCSD(model, test_dataloader,config,loss_func):
    '''
    y = F(x) + lambda_1 * g_1(F(x)_1) + lambda_2 * g_2(F(x)_2), finetuning g_1, g_2, and lambda
    '''
    
    model.eval()
    model.to(config.device)


    FWL = ADCSDModule(output_dim=1,
                        output_window=config.pred_len,
                        num_nodes=config.enc_in,
                        moving_avg=5).to(config.device)
    optimizer = torch.optim.Adam(FWL.parameters(), lr=0.0001, eps=1.0e-8, weight_decay=0, amsgrad=False)

    data_number = 0
    y_truths = []
    y_preds = []
   
 
    output_=[]
    for batch,y,_,_ in tqdm(test_dataloader,total=len(test_dataloader)):
        batch=batch.to(config.device)
        y=y.to(config.device)
        data_number += 1
        output = model(batch)
        if isinstance(output,dict):
            output=output['pred'].unsqueeze(-1)   #(b,t,n,1)
        else:
            output=output.unsqueeze(-1)   #(b,t,n,1)
       
        with torch.no_grad():
            pred = FWL(output)
        
        output_.append(output.detach().cpu().numpy())
        y_truths.append(y.detach().cpu().numpy())
        y_preds.append(pred[...,0].detach().cpu().numpy())
   
        output_=output_[-100:]
        if np.concatenate(output_[-100:], axis=0).shape[0] > config.batch_size+config.pred_len:
            batch_x = torch.tensor(np.concatenate(output_[-100:], axis=0)[-config.batch_size-config.pred_len:-config.pred_len]).to(config.device)
            batch_y = torch.tensor(np.concatenate(y_truths[-100:], axis=0)[-config.batch_size-config.pred_len:-config.pred_len]).to(config.device)
            pred_=FWL(batch_x)
            optimizer.zero_grad()
            loss = loss_func(batch_y,pred_[...,0])
            loss.backward()
            optimizer.step()
    mae=np.mean(np.abs(np.concatenate(y_preds) - np.concatenate(y_truths)))
    mse=np.mean(np.square(np.concatenate(y_preds) - np.concatenate(y_truths)))
    print('mae:',mae)
    print('mse:',mse)
    res={'mae':mae,'mse':mse,'data':config.data,'pred_len':config.pred_len,'model':config.model,
         'seed':config.seed}
    output_path = "ADCSD_itrans24.csv"
    res=pd.DataFrame(res,index=[0])
    res.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

       
import math
def run(pred_len, data, model, seed=2025,seq_len=96,data_path=None):
    from torch.utils.data import DataLoader
    import os
    from dataloader import DatasetCustom
    # 解析参数（假设已定义）
    args = parse_args(model,data,data_path,seq_len,pred_len,seed)
    args.device = 'cuda:0'
    
    # 创建模型保存路径
    os.makedirs(args.checkpoints, exist_ok=True)
    if model=='TimesNet':
        args.train_epochs=10
        args.learning_rate=0.0001
        args.e_layers=2
        args.d_model=min(max(int(int(2**(math.log(args.enc_in)))/2)*2,32),512)
        args.z_loc=1
        args.z_shape=(args.pred_len + args.seq_len,args.d_model)
    
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
    test_loader = DataLoader(test_set, batch_size=24, shuffle=False)
    
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
    model.to(args.device)
    print(f'成功加载模型: {model_path}')
    ADCSD(model, test_loader, args, loss_func=torch.nn.MSELoss())
    
if __name__ == "__main__":
    import itertools
    import pandas as pd
    # df=pd.read_csv('z_experiment_results_itrans.csv')
    # done_experiments = df[['model','seed','pred_len','data','ogd_step','ema']].drop_duplicates()
    # done_experiments['done']=done_experiments.apply(lambda x: ' '.join(str(x) for x in x.values),axis=1)
    # done_experiments=done_experiments['done'].to_list()
    for model,seed,pred_len,data in itertools.product(['iTransformer'],[2024,2026],[1,24,48],['ETTh1','ETTh2','ETTm1','electricity','ETTm2','traffic','PEMS03',
                                                                                         'PEMS04','weather','PEMS07','PEMS08','solar','exchange']):
            run(model=model,seed=seed,pred_len=pred_len,data=data)
    # else:
    #         print(f'实验 {model} {seed} {pred_len} {data} {ogd_step} {ema} 已完成')