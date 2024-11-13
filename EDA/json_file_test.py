import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def test_json():
    # 디렉토리 경로
    directory_path = '../resource/physionet_mai_json'

    # 파일 예시 
    file_cnt = 0
    max_file = 2

    # json모든 파일 읽기
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                ecg_json = json.load(file)
            
            # 'I' 리드의 데이터를 가져와 mv_unit으로 스케일링하여 플로팅
            data = ecg_json['waveform']['data']['I']
            mv_unit = ecg_json['study']['mv_unit']
            if data and mv_unit:  # 데이터가 비어 있지 않은지 확인
                plt.plot(np.array(data) * mv_unit)
                plt.title(filename)  # 각 파일 이름을 그래프 제목으로 표시
            # plt.show()
            
            # 그래프를 파일로 저장
            plt.savefig(f"{filename}_plot.png")
            plt.close()  # 그래프 창 닫기
            
            file_cnt+=1
            if file_cnt>=max_file: # 두개만 읽어보기
                break
    
if __name__ == "__main__":
    test_json()
    