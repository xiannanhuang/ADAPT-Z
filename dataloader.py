import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from typing import Optional, Tuple, List, Dict, Union
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class DatasetCustom(Dataset):
    """
    A custom PyTorch Dataset for time series data.

    Attributes:
        root_path (str): Root directory of the dataset.
        data (str): Dataset name.
        flag (str): Dataset split ('train', 'val', 'test').
        size (Optional[Tuple[int, int, int]]): Sequence, label, and prediction lengths.
        features (str): Feature type ('S', 'M', 'MS').
        data_path (str): Path to the dataset file.
        target (str): Target column name.
        scale (bool): Whether to scale the data.
        inverse (bool): Whether to use inverse transformation.
        timeenc (int): Time encoding type.
        freq (str): Frequency of the data.
        cols (Optional[List[str]]): Columns to use.
        online (str): Online mode ('full').
    """
    def __init__(self, 
                 root_path: str, 
                 data: str, 
                 flag: str = 'train', 
                 size: Optional[Tuple[int, int, int]] = None, 
                 features: str = 'S', 
                 data_path: str = 'ETTh1.csv', 
                 target: str = 'OT', 
                 scale: bool = True, 
                 inverse: bool = False, 
                 timeenc: int = 0, 
                 freq: str = 'h', 
                 cols: Optional[List[str]] = None, 
                 online: str = 'full') -> None:
        """
        Initializes the Dataset_Custom object.
        """
        # Default sequence, label, and prediction lengths
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # Validate flag
        assert flag in ['train', 'test', 'val'], "Flag must be 'train', 'test', or 'val'."
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Initialize attributes
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.data=data

        # Load raw data
        self.df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.len_df = len(self.df_raw)

        # Define data borders
        self.borders = self._define_borders(data)
        self.__read_data__()

    def _define_borders(self, data: str) -> Dict[str, Dict[str, int]]:
        """
        Defines the start and end indices for train, val, and test splits.

        Args:
            data (str): Dataset name.

        Returns:
            Dict[str, Dict[str, int]]: Borders for each split.
        """
        mapping = {
            # ('ETTh1', 'ETTh2'): {
            #     'start': {'train': 0, 'val': 4 * 30 * 24 - self.seq_len, 'test': 5 * 30 * 24 - self.seq_len},
            #     'end': {'train': 4 * 30 * 24, 'val': 5 * 30 * 24, 'test': 20 * 30 * 24}
            # },
            #  ('ETTm1', 'ETTm2'): {
            #     'start': {'train': 0, 'val': 4 * 30 * 24 * 4 - self.seq_len, 'test': 5 * 30 * 24 * 4 - self.seq_len},
            #     'end': {'train': 4 * 30 * 24 * 4, 'val': 5 * 30 * 24 * 4, 'test': 20 * 30 * 24 * 4}
            # },
            # ('ETTh1', 'ETTh2'): {
            #     'start': {'train': 0, 'val': 12 * 30 * 24 - self.seq_len, 'test': 14 * 30 * 24 - self.seq_len},
            #     'end': {'train': 12 * 30 * 24, 'val': 14 * 30 * 24, 'test': 20 * 30 * 24}
            # },
            # ('ETTm1', 'ETTm2'): {
            #     'start': {'train': 0, 'val': 12 * 30 * 24 * 4 - self.seq_len, 'test': 14 * 30 * 24 * 4 - self.seq_len},
            #     'end': {'train': 12 * 30 * 24 * 4, 'val': 14 * 30 * 24 * 4, 'test': 20 * 30 * 24 * 4}
            # },
            # ('electricity', 'Exchange', 'ILI', 'Weather', 'WTH', 'Traffic'): {
            #     'start': {'train': 0, 'val': int(self.len_df * 0.2), 'test': int(self.len_df * 0.25)},
            #     'end': {'train': int(self.len_df * 0.2), 'val': int(self.len_df * 0.25), 'test': int(self.len_df)}
            # },
              ('electricity', 'exchange', 'ILI', 'weather', 'WTH', 'traffic','ETTh2','ETTm1','ETTm2',
               'ETTh1','PEMS03','PEMS04','PEMS07','PEMS08','solar'): {
                'start': {'train': 0, 'val': int(self.len_df * 0.6), 'test': int(self.len_df * 0.7)},
                'end': {'train': int(self.len_df * 0.6), 'val': int(self.len_df * 0.7), 'test': int(self.len_df)}
            },
            ('nyctaxi','nycbike'):{
                'start': {'train': 0, 'val': 334*24, 'test': 365*24,},
                'end': {'train': 334*24, 'val':365*24, 'test': int(self.len_df)}
            }
        }

        borders = {}
        for keys, value in mapping.items():
            for key in keys:
                borders[key] = value
        return borders[data]

    def __read_data__(self) -> None:
        """
        Reads and processes the dataset.
        """
        self.scaler = StandardScaler()
        df_raw = self.df_raw

        border1 = self.borders['start'][self.flag]
        border2 = self.borders['end'][self.flag]

        # Select features
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            if self.data=='nyctaxi':
                cols_data = [str(i) for i in range(224)]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale data
        if self.scale:
            train_data = df_data[self.borders['start']['train']:self.borders['end']['train']]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time encoding
        if self.timeenc == 2:
            train_df_stamp = df_raw[['date']][self.borders['start']['train']:self.borders['end']['train']]
            train_df_stamp['date'] = pd.to_datetime(train_df_stamp.date)
            train_date_stamp = time_features(train_df_stamp, timeenc=self.timeenc)
            date_scaler = sklearn_StandardScaler().fit(train_date_stamp)

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            data_stamp = date_scaler.transform(data_stamp)
        else:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Assign processed data
        self.data_x = data[border1:border2]
        self.data_y = df_data.values[border1:border2] if self.inverse else data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves a single data sample.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Input sequence, target sequence, 
            input time features, and target time features.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len], 
                                    self.data_y[r_begin + self.label_len:r_end]], axis=0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x=torch.tensor(seq_x, dtype=torch.float32)
        seq_y=torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark=torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark=torch.tensor(seq_y_mark, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies inverse transformation to the data.

        Args:
            data (np.ndarray): Data to inverse transform.

        Returns:
            np.ndarray: Inverse transformed data.
        """
        return self.scaler.inverse_transform(data)
