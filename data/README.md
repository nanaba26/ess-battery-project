# data

원본 데이터는 용량(약 8GB) 문제로 저장소에 포함하지 않습니다.

## 다운로드
- Kaggle : https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 원본 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)

## 배치
아래 파일을 이 폴더(`data/`)에 그대로 넣습니다.

| 파일 | Batch | 용도 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (2.8GB) | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (1.9GB) | Batch 2 | 테스트 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (3.0GB) | Batch 3 | 추가 검증 |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` (0.1GB) | extra | 미사용 (다른 논문 실험) |

이후 `python src/features.py` 를 실행하면 `results/features.csv` 가 생성됩니다.
