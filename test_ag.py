import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader.data_loader import create_dataloaders
from model_ag import ResNetECG

def test(model, dataloader, device, log_file="log/test_ag_log_4_model_6.txt"):
    model.eval()
    all_labels = []
    all_outputs = []
    total_accuracy, total_f1, total_precision, total_recall = 0, 0, 0, 0
    num_batches = 0
    
    with open(log_file, "a") as f:
        with torch.no_grad():
            for batch in dataloader:
                # 각 배치에서 age, gender, data, labels를 언패킹
                age, gender, data, labels = batch
                data, labels = data.unsqueeze(2).to(device), labels.to(device)  # data의 형태 [batch_size, 12, 1, 5000]
                age_gender = torch.stack([age, gender], dim=1).to(device)  # [batch_size, 2] 형태로 만듦
                
                # 모델 예측
                outputs = model(data, age_gender)
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
                
                # 배치별로 평가 지표 계산 (단일 클래스일 경우)
                if len(torch.unique(labels)) <= 1:
                    binary_outputs = (outputs >= 0.5).float()  # 임계값 0.5로 이진화
                    total_accuracy += accuracy_score(labels.cpu(), binary_outputs)
                    total_f1 += f1_score(labels.cpu(), binary_outputs, average="macro")
                    total_precision += precision_score(labels.cpu(), binary_outputs, average="macro", zero_division=1)
                    total_recall += recall_score(labels.cpu(), binary_outputs, average="macro", zero_division=1)
                    num_batches += 1

        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)
        
        # 다중 클래스일 경우 AUROC 및 AUPRC 계산
        if len(torch.unique(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_outputs, average="macro", multi_class="ovr")
            auprc = average_precision_score(all_labels, all_outputs, average="macro")
            f.write(f"Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}\n")
            return auroc, auprc
        else:
            # 단일 클래스일 경우 누적된 지표 평균 계산
            avg_accuracy = total_accuracy / num_batches
            avg_f1 = total_f1 / num_batches
            avg_precision = total_precision / num_batches
            avg_recall = total_recall / num_batches
            f.write(f"Test Evaluation (Single Class) - Accuracy: {avg_accuracy:.4f}, F1 Score: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}\n")
            return None, None  # AUROC와 AUPRC는 단일 클래스에서 None으로 설정

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ResNetECG(num_classes=26).to(device)
    model.load_state_dict(torch.load("models/model_epoch_6.pth"))  # 학습된 모델 로드
    
    # 로그 파일 경로 설정
    log_file = "log/test_ag_log_4_model_6.txt"
    
    _, test_loader = create_dataloaders('../resource/physionet2021_total.csv', '../resource/physionet_mai_json')

    test_auroc, test_auprc = test(model, test_loader, device)

    with open(log_file, "a") as f:
        if test_auroc and test_auprc:
            f.write(f"Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}\n")

if __name__ == "__main__":
    main()
