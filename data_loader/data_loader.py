import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import random

class ECGDataset(Dataset):
    def __init__(self, csv_file, json_dir, transform=None, is_test=False, target_length=5000):
        # CSV 파일 불러오기
        self.data_table = pd.read_csv(csv_file)
        # fold == 4인 데이터를 테스트셋으로 분리
        self.data_table = self.data_table[self.data_table['fold'] == 4] if is_test else self.data_table[self.data_table['fold'] != 4]
        self.json_dir = json_dir  # JSON 파일 경로
        self.transform = transform
        self.target_length = target_length  # 고정된 길이 설정

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        # CSV에서 objectid 가져오기
        row = self.data_table.iloc[idx]
        objectid = row['objectid']
        json_path = os.path.join(self.json_dir, f"{objectid}.json")

        # JSON 파일이 없는 경우 예외 처리
        if not os.path.exists(json_path):
            return None  # 파일이 없는 경우 None 반환
        
        # JSON 파일 불러오기
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        
        # JSON에서 필요한 데이터 추출 (결측값 처리 추가)
        age = json_data['person'].get('age', 59)  # age가 None이면 59로 설정
        if age is None:
            age = 59
        gender = json_data['person'].get('gender')
        if gender is None:
            gender = random.choice([-1, 1])  # gender가 None이면 -1 또는 1 중 랜덤 할당
        
        # (1,5000) 또는 다양한 길이의 데이터를 (12, target_length)로 변환
        lead_data = []
        for lead in json_data['waveform']['data'].values():
            lead_tensor = torch.tensor(lead, dtype=torch.float32)
            # 길이 맞추기 (잘라내기 또는 패딩)
            if lead_tensor.size(0) > self.target_length:
                lead_tensor = lead_tensor[:self.target_length]  # 길면 자르기
            else:
                padding = self.target_length - lead_tensor.size(0)
                lead_tensor = torch.cat([lead_tensor, torch.zeros(padding)], dim=0)  # 짧으면 패딩
            lead_data.append(lead_tensor)
        
        lead_data = torch.stack(lead_data)  # (12, target_length) 형식
        
        # CSV 파일에서 라벨 정보 가져오기 (3열부터 29열까지 읽기)
        labels = pd.to_numeric(row[3:29], errors='coerce').fillna(0).astype(float).values
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            lead_data = self.transform(lead_data)

        # `age`, `gender`, `lead_data`, `labels` 순서로 반환
        return age, gender, lead_data, labels

def collate_fn(batch):
    # None 데이터를 제외하고 유효한 데이터만 반환
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # 빈 배치를 None으로 반환하여, DataLoader에서 건너뛸 수 있도록 처리
    return torch.utils.data.dataloader.default_collate(batch)


# 학습 및 테스트 데이터 로더 생성
# batch_size = 64, num_workers = 4 이었음
def create_dataloaders(csv_file, json_dir, batch_size=64, num_workers=0, target_length=5000):
    # 학습 데이터 로더
    train_dataset = ECGDataset(csv_file=csv_file, json_dir=json_dir, is_test=False, target_length=target_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)

    # 테스트 데이터 로더
    test_dataset = ECGDataset(csv_file=csv_file, json_dir=json_dir, is_test=True, target_length=target_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)

    return train_loader, test_loader
