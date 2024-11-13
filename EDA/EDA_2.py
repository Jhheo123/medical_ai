import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# JSON 디렉터리에서 age와 gender 데이터를 추출하여 CSV로 저장하는 함수
def extract_age_gender(json_dir, output_csv='EDA_result/age_gender_data.csv'):
    data_list = []
    missing_count = 0  # 누락된 데이터 수를 세기 위한 변수
    
    print("나이 및 성별 데이터 추출 중")
    for filename in tqdm(os.listdir(json_dir), desc="나이 및 성별 데이터 추출"):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            
            with open(json_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    
                    # 기본값 설정 및 로그 추가
                    age = json_data.get('person', {}).get('age', None)
                    gender = json_data.get('person', {}).get('gender', None)

                    if age is None or gender is None:
                        missing_count += 1  # 누락된 데이터가 있을 때 카운트를 증가
                        print(f"경고: 파일 {filename}에서 나이 또는 성별 누락")
                    
                    objectid = os.path.splitext(filename)[0]
                    
                    data_list.append({"objectid": objectid, "age": age, "gender": gender})
                
                except json.JSONDecodeError:
                    print(f"오류: JSON 파일 해석 실패: {filename}")
    
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"나이 및 성별 데이터가 {output_csv}에 저장되었습니다.")
    print(f"총 누락 항목 수 (나이 또는 성별): {missing_count}")
    return df

# 결측치 처리 및 통계 분석 함수
def analyze_age_gender_data(df, output_path="EDA_result"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 결측치 처리
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['gender'].fillna(df['gender'].mode()[0], inplace=True)
    
    # 결측치 처리 후 데이터 저장
    df.to_csv(os.path.join(output_path, "processed_age_gender_data.csv"), index=False)
    print(f"처리된 나이 및 성별 데이터가 {os.path.join(output_path, 'processed_age_gender_data.csv')}에 저장되었습니다.")

    # 기본 통계 정보
    with open(os.path.join(output_path, "age_gender_statistics.txt"), "w") as f:
        f.write("Basic Statistics:\n")
        f.write(df.describe().to_string())
        f.write("\n\nMissing Values:\n")
        f.write(df.isnull().sum().to_string())
    
    print("기본 통계 및 누락 값 정보가 age_gender_statistics.txt에 저장되었습니다.")

    # 나이 분포 시각화
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_path, "age_distribution.png"))
    plt.close()
    print(f"나이 분포 그래프가 {os.path.join(output_path, 'age_distribution.png')}에 저장되었습니다.")

    # 성별 분포 시각화
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='gender')
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_path, "gender_distribution.png"))
    plt.close()
    print(f"성별 분포 그래프가 {os.path.join(output_path, 'gender_distribution.png')}에 저장되었습니다.")

# 메인 함수
def main():
    json_dir = '../resource/physionet_mai_json'
    output_csv = "EDA_result/age_gender_data.csv"
    
    # JSON 파일에서 나이 및 성별 데이터를 추출하고 CSV로 저장
    df = extract_age_gender(json_dir, output_csv)
    
    # 추출된 데이터를 로드하여 결측값을 처리하고 분석
    analyze_age_gender_data(df)

if __name__ == "__main__":
    main()
