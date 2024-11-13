# medical_ai
### 최종 성능 결과
- AUROC: 0.9371, AUPRC: 0.5429 (model_epoch_7.pth 기준)
```bash
workspace/log/test_ag_log_4_model_7.txt
```
- weight 값 위치
```bash
workspace/models_weight/model_epoch_7.pth
```
### 모델 학습 및 평가 결과 산출 재현 방법
```bash
conda activate resnet
cd workspace
```
- train
```bash
python train_ag.py
```
  - train 관련 log 파일
```bash
workspace/log/train_log_4_age_gender_test.txt
```  
- inference  
```bash
python test_ag.py
```
  - inference 관련 log 파일
```bash
workspace/log/test_ag_log_4_model_6.txt
```
### EDA 관련 파일
```bash
workspace/EDA/eda_df.ipynb
```
- 관련 산출물
```bash
workspace/EDA/EDA_result
```
### Data Loader
- fold 0-3: train/validation dataset
- fold 4: test dataset
- lead 길이: 5000
- 결측값 처리
  - age: 59 (평균)
  - gender: -1(여성),1(남성) 랜덤 할당 
