import os
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from data_loader.data_loader import ECGDataset, collate_fn  # 수정된 collate_fn 사용
from model_ag import ResNetECG
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm  # tqdm 모듈 추가
import random
from sklearn.model_selection import StratifiedShuffleSplit

# 모델 저장 및 로그 폴더 생성
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("log"):
    os.makedirs("log")

def stratified_sample(dataset, sample_size):
    # None 값이 아닌 유효한 레코드들만 추출
    valid_indices = [i for i in range(len(dataset)) if dataset[i] is not None]
    labels = [dataset[i][3].argmax().item() for i in valid_indices]
    
    # 각 클래스별로 stratified sampling
    stratified_indices = []
    for label in set(labels):
        class_indices = [idx for idx, l in zip(valid_indices, labels) if l == label]
        class_sample = random.sample(class_indices, min(len(class_indices), sample_size // len(set(labels))))
        stratified_indices.extend(class_sample)
    
    return torch.utils.data.Subset(dataset, stratified_indices)

def test_model(model, input_shape=(64, 12, 1, 5000), device="cpu"):  # (64, 12, 1, 5000) 형태로 변경
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)  # # [batch_size, 12, 1, 5000] 형태 유지

    # age_gender의 더미 데이터 생성 [batch_size, 2]
    age_gender_dummy = torch.randn(input_shape[0], 2).to(device)  # batch_size 크기와 일치하는 [64, 2] 더미 생성

    with torch.no_grad():
        output = model(dummy_input, age_gender_dummy)  # age_gender 추가
    print(f"모델 출력 차원: {output.shape}")



def split_train_valid(train_loader, valid_ratio=0.2):
    dataset = train_loader.dataset
    train_size = int(len(dataset) * (1 - valid_ratio))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers, collate_fn=collate_fn)
    return train_loader, valid_loader

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_file="log/train_log_2.txt"):
    model.train()
    running_loss = 0.0

    with open(log_file, "a") as f:
        f.write(f"\nEpoch {epoch} 시작\n")
        
        batch_iterator = tqdm(dataloader, desc="Training", unit="batch")
        for batch_idx, batch in enumerate(batch_iterator):
            if batch is None:  # 빈 배치일 경우 건너뜀
                f.write(f"빈 배치가 {batch_idx}번째 인덱스에서 발견되었습니다.\n")
                continue
            
            try:
                age, gender, data, labels = batch  # 새로운 순서로 언패킹
                data = data.unsqueeze(2).to(device)  # [batch_size, 12, 1, 5000] 형태로 조정
                # data = data.to(device)  # unsqueeze(2) 제거하여 [batch_size, 12, 5000] 형식 유지
                labels = labels.to(device)
                
                # 나이와 성별 정보 결합
                age_gender = torch.stack([age, gender], dim=1).to(device) # [batch_size, 2]
                
                optimizer.zero_grad()
                outputs = model(data, age_gender)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * data.size(0)
                
                batch_iterator.set_postfix(loss=loss.item())
                f.write(f"Batch [{batch_idx + 1}/{len(dataloader)}], 손실: {loss.item():.4f}\n")
            except Exception as e:
                f.write(f"{batch_idx}번째 배치에서 에러 발생: {e}\n")
                f.write(f"배치 내용: {batch}\n")
                continue

        epoch_loss = running_loss / len(dataloader.dataset)
        f.write(f"Epoch {epoch} 손실: {epoch_loss:.4f}\n")
    
    return epoch_loss


def evaluate(model, dataloader, device, log_file="log/train_log_2.txt"):
    model.eval()
    all_labels = []
    all_outputs = []
    with open(log_file, "a") as f:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch is None:  # 빈 배치 스킵
                    f.write(f"{batch_idx}번째 배치가 비어 있어 스킵합니다.\n")
                    continue
                
                # 정상적인 배치일 때 처리
                age, gender, data, labels = batch  # 새로운 순서로 언패킹
                data = data.unsqueeze(2).to(device)  # [batch_size, 12, 1, 5000] 형태로 조정
                # data = data.to(device)  # unsqueeze(2) 제거하여 [batch_size, 12, 5000] 형식 유지
                labels = labels.to(device)

                # 나이와 성별 정보 결합
                age_gender = torch.stack([age, gender], dim=1).to(device) # [batch_size, 2]
                
                outputs = model(data, age_gender)
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
                
            all_labels = torch.cat(all_labels)
            all_outputs = torch.cat(all_outputs)
            # 단일 클래스 여부를 확인하여 적절한 지표 계산
            if len(torch.unique(all_labels)) and len(torch.unique(all_outputs))> 1:
                # 여러 클래스일 경우 AUROC 및 AUPRC 계산
                auroc = roc_auc_score(all_labels, all_outputs, average="macro", multi_class="ovr")
                auprc = average_precision_score(all_labels, all_outputs, average="macro")
                f.write(f"Evaluation - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}\n")
            else:
                # 단일 클래스일 경우 F1 점수, 정밀도, 재현율 사용
                binary_outputs = (all_outputs >= 0.5).float()  # 임계값 0.5로 이진화
                accuracy = accuracy_score(all_labels, binary_outputs)
                f1 = f1_score(all_labels, binary_outputs, average="macro")
                precision = precision_score(all_labels, binary_outputs, average="macro", zero_division=1)
                recall = recall_score(all_labels, binary_outputs, average="macro", zero_division=1)
                f.write(f"Evaluation (Single Class) - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
                auroc, auprc = None, None  # AUROC와 AUPRC는 단일 클래스에서 None으로 설정

    return auroc, auprc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetECG(num_classes=26).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label
    optimizer = Adam(model.parameters(), lr=0.003)

    # 모델 테스트 실행
    test_model(model, input_shape=(64, 12, 1, 5000), device=device)
    
    # 학습 및 테스트 데이터 로더 생성
    csv_file_path = '../resource/physionet2021_total.csv'
    json_dir_path = '../resource/physionet_mai_json'
    train_dataset = ECGDataset(csv_file=csv_file_path, json_dir=json_dir_path, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # train/validation 분할
    train_loader, valid_loader = split_train_valid(train_loader)
    
    # 여러 클래스를 포함한 샘플링된 데이터셋 생성 (테스트용)
    # train_subset_size = 100
    # valid_subset_size = 100
    
    # train_subset_indices = random.sample(range(len(train_loader.dataset)), train_subset_size)
    # valid_subset_indices = random.sample(range(len(valid_loader.dataset)), valid_subset_size)
    
    # train_subset = torch.utils.data.Subset(train_loader.dataset, train_subset_indices)
    # valid_subset = torch.utils.data.Subset(valid_loader.dataset, valid_subset_indices)
    
    # train_loader_small = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # valid_loader_small = DataLoader(valid_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 다양한 클래스 포함을 위한 stratified 샘플링
    train_subset = stratified_sample(train_loader.dataset, 100)
    valid_subset = stratified_sample(valid_loader.dataset, 100)
    
    train_loader_small = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    valid_loader_small = DataLoader(valid_subset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 로그 파일 초기화
    log_file = "log/train_log_4_age_gender_test.txt"
    with open(log_file, "w") as f:
        f.write("훈련 로그 시작\n")
        
    # 작은 데이터셋으로 테스트
    print("작은 데이터셋으로 테스트 시작")
    train_loss = train_epoch(model, train_loader_small, criterion, optimizer, device, epoch=1, log_file=log_file)
    val_auroc, val_auprc = evaluate(model, valid_loader_small, device, log_file=log_file)
    print(f"테스트 완료 - Train Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}\n")
    
    # 에폭 반복
    for epoch in range(1, 11):  # 10
        with open(log_file, "a") as f:
            f.write(f"\nEpoch [{epoch}/10]\n")
        
        # 에폭 별 훈련 실행 및 손실 기록
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_file=log_file)
        
        # 검증 데이터셋 평가
        val_auroc, val_auprc = evaluate(model, valid_loader, device, log_file=log_file)
        
        with open(log_file, "a") as f:
            f.write(f"Epoch [{epoch}/10] 종료 - Train Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}\n")

        # 모델 저장
        torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
