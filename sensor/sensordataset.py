import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class SensorDataset(Dataset):
    def __init__(self, input_dir, data_types, transform=None):
        self.input_dir = input_dir
        self.data_types = data_types
        self.transform = transform

        self.body_parts = ['Pelvis', 'Head', 'Left Forearm', 'Left Lower Leg', 'Left Shoulder',
                           'Left Upper Arm', 'Left Upper Leg', 'Right Forearm', 'Right Lower Leg',
                           'Right Shoulder', 'Right Upper Arm','Right Upper Leg']

        self.csv_files = []
        self.labels = []

        input_n_dir = os.path.join(self.input_dir, 'N', 'N')
        for csv_folder in os.listdir(input_n_dir):
            csv_folder_path = os.path.join(input_n_dir, csv_folder)
            if os.path.isdir(csv_folder_path):
                for file_name in os.listdir(csv_folder_path):
                    if file_name.endswith('.csv'):
                        self.csv_files.append(os.path.join(csv_folder_path, file_name))
                        self.labels.append(0)  

        input_y_dir = os.path.join(self.input_dir, 'Y')
        for sub_group in ['BY', 'FY', 'SY']:
            input_sub_dir = os.path.join(input_y_dir, sub_group)
            for csv_folder in os.listdir(input_sub_dir):
                csv_folder_path = os.path.join(input_sub_dir, csv_folder)
                if os.path.isdir(csv_folder_path):
                    for file_name in os.listdir(csv_folder_path):
                        if file_name.endswith('.csv'):
                            self.csv_files.append(os.path.join(csv_folder_path, file_name))
                            self.labels.append(1) 

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        label = self.labels[idx]
        df = pd.read_csv(csv_file)

        body_part_tensors = []

        for body_part in self.body_parts:
            part_cols = []
            for data_type in self.data_types:
                part_cols += [col for col in df.columns if data_type in col and body_part in col]

            part_data = df[part_cols].values

            part_data_avg = part_data.reshape(10, 60, -1).mean(axis=1)

            body_part_tensor = torch.tensor(part_data_avg, dtype=torch.float32)
            body_part_tensors.append(body_part_tensor)

        sample = torch.stack(body_part_tensors)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

