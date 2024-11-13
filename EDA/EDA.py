import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_loader import ECGDataset
import random

# EDA 결과 폴더 생성 함수
def ensure_eda_result_dir():
    if not os.path.exists("EDA_result"):
        os.makedirs("EDA_result")


# 데이터 로드 함수 정의
def load_eda_data(csv_file, json_dir, log_file="EDA_result/eda_log_print.txt"):
    data_list = []
    dataset = ECGDataset(csv_file=csv_file, json_dir=json_dir)
    
    # 몇 개 샘플을 출력하여 구조를 확인
    for idx in range(3):  # 처음 몇 개 데이터만 출력해 봅니다.
        sample = dataset[idx]
        print(f"Sample {idx}: {sample}")
    
    # 로그 파일 초기화
    if not os.path.exists("EDA_result"):
        os.makedirs("EDA_result")
    
    with open(log_file, "w") as log:
        log.write("데이터 로딩 시작\n")

    print("데이터 로딩 시작")
    for idx in tqdm(range(len(dataset)), desc="Loading Data"):
        sample = dataset[idx]
        
        with open(log_file, "a") as log:
            if sample is None:
                log.write(f"Warning: Sample at index {idx} is None. Skipping.\n")
                continue
            
            # 새로운 반환 순서에 맞춰서 데이터를 언패킹
            age, gender, lead_data, labels = sample
            lead_data_mean = lead_data.mean().item()
            lead_data_std = lead_data.std().item()
            
            # age 결측값 처리 및 로그
            if pd.isnull(age):
                log.write(f"Missing age at index {idx}. Setting default age to 59.\n")
                age = 59  # 기본값
            else:
                log.write(f"Index {idx}: Age is {age}, no missing value.\n")
            
            # gender 결측값 처리 및 로그
            if pd.isnull(gender):
                log.write(f"Missing gender at index {idx}. Assigning random gender (-1 or 1).\n")
                gender = random.choice([-1, 1])
            else:
                log.write(f"Index {idx}: Gender is {gender}, no missing value.\n")
            
            # 데이터를 리스트에 추가
            data_list.append({
                "lead_data_mean": lead_data_mean,
                "lead_data_std": lead_data_std,
                "labels": labels.numpy(),
                "age": age,
                "gender": gender
            })

    # 최종 데이터프레임 생성 및 로그 기록
    eda_df = pd.DataFrame(data_list)
    with open(log_file, "a") as log:
        log.write("데이터 로딩 완료\n")
        log.write(f"총 {len(data_list)}개의 데이터가 로드되었습니다.\n")
    
    print("데이터 로딩 완료")
    return eda_df


# 기본 통계 정보 출력 및 저장
def print_basic_statistics(eda_df, log_file="EDA_result/eda_log_distribution_1.txt"):
    ensure_eda_result_dir()  # 결과 폴더 확인 및 생성
    with open(log_file, "w") as f:
        f.write("Basic statistics of the dataset:\n")
        f.write(eda_df.describe().to_string())
        f.write("\n\nMissing values check:\n")
        f.write(eda_df.isnull().sum().to_string())
    print("기본 통계 정보와 결측값 확인이 eda_log.txt에 저장되었습니다.")

# 라벨 분포 시각화
def plot_label_distribution(eda_df, save_path="EDA_result/label_distribution_1.png"):
    ensure_eda_result_dir()
    label_counts = pd.DataFrame(eda_df['labels'].tolist()).sum(axis=0)
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.title("Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()
    print(f"라벨 분포 그래프가 {save_path}에 저장되었습니다.")

# 리드 데이터 평균 및 표준편차 분포 시각화
def plot_lead_data_statistics(eda_df, save_path_mean="EDA_result/lead_data_mean_distribution_1.png", save_path_std="EDA_result/lead_data_std_distribution_1.png"):
    ensure_eda_result_dir()
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(eda_df['lead_data_mean'], bins=30, alpha=0.7)
    plt.title('Lead Data Mean Distribution')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.savefig(save_path_mean)
    plt.close()
    
    plt.figure(figsize=(6, 5))
    plt.hist(eda_df['lead_data_std'], bins=30, alpha=0.7)
    plt.title('Lead Data Standard Deviation Distribution')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.savefig(save_path_std)
    plt.close()
    
    print(f"리드 데이터 평균 분포 그래프가 {save_path_mean}에 저장되었습니다.")
    print(f"리드 데이터 표준편차 분포 그래프가 {save_path_std}에 저장되었습니다.")

# 나이 및 성별에 따른 라벨 분포 분석
def analyze_age_gender(eda_df, save_path_age_gender="EDA_result/age_gender_scatter_1.png", save_path_gender_label="EDA_result/gender_label_distribution_1.png"):
    ensure_eda_result_dir()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=eda_df['age'], y=eda_df['lead_data_mean'], hue=eda_df['gender'])
    plt.title("Average Lead Data by Age (Colored by Gender)")
    plt.xlabel("Age")
    plt.ylabel("Average Lead Data")
    plt.legend(title="Gender")
    plt.savefig(save_path_age_gender)
    plt.close()
    print(f"나이와 성별에 따른 평균 리드 데이터 산점도가 {save_path_age_gender}에 저장되었습니다.")
    
    # 성별에 따른 라벨 분포
    gender_labels = pd.DataFrame(list(eda_df['labels']), index=eda_df['gender']).groupby(level=0).sum()
    gender_labels.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Label Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Label Count")
    plt.savefig(save_path_gender_label)
    plt.close()
    print(f"성별에 따른 라벨 분포 그래프가 {save_path_gender_label}에 저장되었습니다.")

# eda_df를 저장
def save_eda_df(eda_df, csv_path="EDA_result/eda_df.csv", pkl_path="EDA_result/eda_df.pkl"):
    # 결과 폴더 확인 및 생성
    ensure_eda_result_dir()
    
    # CSV 파일로 저장
    eda_df.to_csv(csv_path, index=False)
    print(f"EDA DataFrame이 CSV 파일로 {csv_path}에 저장되었습니다.")
    
    # PKL 파일로 저장
    eda_df.to_pickle(pkl_path)
    print(f"EDA DataFrame이 PKL 파일로 {pkl_path}에 저장되었습니다.")

# 메인 함수
def main():
    csv_file = '../resource/physionet2021_total.csv'
    json_dir = '../resource/physionet_mai_json'
    
    eda_df = load_eda_data(csv_file, json_dir)
    # eda_df 저장
    save_eda_df(eda_df)
    
    # print_basic_statistics(eda_df)
    # plot_label_distribution(eda_df)
    # plot_lead_data_statistics(eda_df)
    # analyze_age_gender(eda_df)

if __name__ == "__main__":
    main()
