## level2-klue-nlp-01

```
|-- Dockerfile
|-- data
|   |-- dev-v.0.0.2.csv
|   |-- dict_label_to_num.pkl
|   |-- dict_num_to_label.pkl
|   |-- test.csv
|   |-- train-v.0.0.0.csv
|   `-- train-v.0.0.2.csv
|-- data_utils
|   |-- __init__.py
|   `-- data_utils.py
|-- metrics
|   |-- __init__.py
|   `-- metrics.py
|-- parameters
|   `-- roberta-small-128-5e-05.pt
|-- results
|   `-- roberta-submission.pt
`-- utils
|   |-- __init__.py
|   |-- preprocessing.py
|   `-- utils.py
|-- inference.py
|-- requirements.txt
|-- settings.py
|-- train.py
```


### Docker setting
**1.clone this repository**
``` 
git clone https://github.com/boostcampaitech6/level2-klue-nlp-01.git
cd level2-klue-nlp-01
```

**2.build Dockerfile**
```
docker build --tag [filename]:1.0
```

**3.execute**

```
# Docker version 2.0 or later.
docker run -itd --runtime=nvidia --name dgl_tuto -p 8888:8888 -v C:\Users\Name\:/workspace [filename]:1.0 /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name boostcamp -p 8888:8888 -v C:\Users\Name\:/workspace [filename]:1.0 /bin/bash
```
  
**4.use jupyter notebook**
```
docker exec -it boostcamp bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```


# 0. 프로젝트 요약

- **전처리** 
    - Typed entity marker(punct)
    - UNK(언노운 토큰) Vocab 처리
    - 한자->한글 번역
    - Task-adaptive pretraining
    - Imbalanced Data Sampler
    - Ensemble(weight sum)

- **하이퍼 파라미터**
    - Loss function
    (Label Smoothing Loss)
    - Max Length (128)
    - Scheduler (CosineAnnealingLR)
    - Batch Size(64)
    - Learning Rate(3e-5)

- **최종 점수**
    - F1 Score
    (75.2514)
    - AUPCR Score
    (82.1402)

# 1. 프로젝트 개요

### 1.1 프로젝트 주제

- Relation Extraction은 subject entity와 object entity의 관계를 파악해서, 올바른 클래스를 예측하는 것을 목표로 한다.
- RE는 QA 구축 및 언어 정보를 바탕으로 한 서비스 등에 사용하며, 본 대회에선 30개의 클래스 라벨에 대한 각각의 확률을 예측한다.

### 1.2 프로젝트 구현 내용

- RE 프로젝트를 통해 모델이 단어의 속성과 관계를 어떻게 파악하는지 알아본다.
- 프로젝트에서 전처리, 데이터 증강, pre-trained 모델을 활용하여 각각의 자연어 처리 성능을 비교해볼 수 있다.

### 1.3 활용 장비 및 재료(개발 환경, 협업 tool 등)

- VS Code + SSH 접속을 통해 AI stage 서버 GPU 활용
- Git을 통한 버전 관리, WandB로 최적 파라미터 및 실험 결과 저장, Github를 통한 코드 공유 및 모듈화
- Slack, Zoom, Notion을 통한 프로젝트 일정 및 문제 상황 공유 + 회의 진행

### 1.4 데이터셋 설명

- 기본제공 데이터셋 train.csv, test_data.csv
- 위키트리, 위키피디아, 정책브리핑 총 3가지 출처의 문장 데이터로 구성
- train 총 32740 행, test_data 총 7765 행
- 이후 대회 진행 중 전처리 및 증강에 따라 학습을 진행하는 데이터를 변경하여 사용함.

# 2. 프로젝트 팀 구성 및 역할

- 김인수(팀장) : EDA, 데이터 전처리, Loss Function 구현, entity marker 구현, BiLSTM Layer 추가
- 김동언(팀원) : EDA, 한자 → 한글 번역, UNK Token 구축, 앙상블 구축, 코드 모듈화, 하이퍼파라미터 튜닝
- 오수종(팀원) : EDA, 이진 및 다진 분류 코드 구현, 클래스 가중치 loss구현
- 이재형(팀원) : EDA, 전처리, Loss Function 구현
- 임은형(팀원) : EDA, 데이터 전처리, 증강, ImbalancedDatasetSampler, confusion matrix 시각화
- 이건하(팀원) : EDA, 일반화 코드 초안 정리, TAPT

# 3. 프로젝트 수행 절차 및 방법

### 3.1 팀 목표 설정

- 01/03~01/05 : Baseline코드 실습, EDA, Model 리서치
- 01/08~01/12 : Data PreProcessing, 모델 modification(TPAT, biLSTM Layer)
- 01/15~01/19 : Data Augmentation, Hyperparameter 튜닝, Ensemble 수립

### 3.2 프로젝트 사전기획

- 역할 분담 : 모더레이터(시간 분배 및 회의 진행), 스터디담당(논문 스터디 진행 및 관리), 일정담당(일정 리마인더), 대회담당(대회 일정 및 방향 제시), 깃헙담당(깃헙 관리 및 컨벤션 제시), 서기(각종 기록 취합 및 제출)
- 이외에 대회에 필요한 태스크 분배해서 진행
- Notion : 조사 및 실험 결과 공유, 기록 정리 및 보관
- Github: 공용 코드 정리, 모듈화 및 코드리뷰, 진행사항 공유
- Zoom : 화상 회의를 통한 실시간 상황 공유, 결과 피드백, 이슈 해결

### 3.3 프로젝트 수행

- Project Pipeline
    
    ![Project Pipeline](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/01575992-d99d-46d4-81c2-266cccdee078)

    
    1. 프로젝트 개발환경 구축(Github, Slack, WandB, Server)
    2. Baseline 코드 분석 및 EDA를 통한 데이터 구조 파악
    3. Relation Extraction Task 이해, 기법 조사
    4. Baseline model 리서치 및 탐색(klue/roberta-large, kykim/electra 등)
    5. 데이터 EDA, 데이터 전처리(불용어제거, 정규식, 띄어쓰기, 맞춤법, 오타, UNK토큰 등) 전략 수립
    6. 데이터 증강으로 라벨의 분포 일정하게 생성 (도치, 역번역, label swap)
    7. WandB Sweep을 이용하여 Hyper Parameters Tuning 및 기법 실험
    8. Early Stopping을 이용하여 과적합 및 시간 낭비 방지
    9. Soft Voting을 진행하여 최종 결과물 제작

# 4. 프로젝트 수행 결과

- 아래 사항들 중 최종 모델에 적용된 것들은 [최종] 태그 추가함.

  ![Time Table](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/37e8f481-67e8-4f2a-a937-8c4d4f73fd5b)


### 4.1 Study

- **RE Task**
    - 목표: RE에 대한 다양한 구현코드, 데이터셋을 찾아보면서 태스크에 대한 이해도를 높이고자 함
    - 결과: Raw sentence뿐만 아니라 type과 같은 side information에 따라 성능이 달라질 수 있다는 점 확인

### 4.2 EDA

- Sentence Evaluation
  ![Sentence Evaluation](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/25de87ec-1132-4692-b143-511a1da51496)

- Train data label
  ![Train data label](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/45ac4213-c7e9-4214-8326-929b3d81aa08)


- subject entity, object entity
  - subject entity type
  <img width="300" alt="subject entity1" src="https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/6a348798-d2f6-44c2-b54e-aa9d858182c8">
  <img width="450" alt="subject entity2" src="https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/471049b8-fda8-4589-a5bf-40b145d503c7">

  - object entity type
  <img width="300" alt="object entity1" src="https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/91684ca1-80ee-47e4-aa45-069dcae4dbcc">
  <img width="450" alt="object entity2" src="https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/b651be77-a953-40aa-bc93-30cbe1c5ea4f">

- **Target 값 분포**
    - train data label 분포에서 no_relation이 많았으며 train data의 분포 그대로 train data의 일부를validation data로 사용하였다.
    
    → Train 데이터셋의 data imbalace를 줄이기 위한 다양한 시도 진행
    
- **Source 별 분포**
    - Train 데이터셋의 Source별 분포 파악한 결과, Wikipedia가 압도적으로 많음.
    - 특히 wikitree 소스는 org:top_members/employee 라벨에 과도하게 편중됨
    - Test 데이터셋은 Wikipedia와 Wikitree의 수가 거의 비슷

  ![Source 별 분포](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/61376944-2515-4d32-85cf-7542c190a57b)


### 4.3 Preprocessing

- **Typed entity marker(punct)[최종]**
    - 이유: 모델이 문장을 학습할 때 subject,object entity를 잘 파악할 수 있게 할 수 없을까?
    → 기존 문장에서 존재하지 않는 특수기호들을 삽입함으로써 entity들을 구분하여 학습할 수 있게 함.
    - 구현 : sentence에 subject,object entity 앞뒤로 “@*”, “#^” 삽입하여 변환, 
    “@*{subject_type}*{subject_name}@와 #^{object_type}^{object_name}# 의 관계” + sentence를 tokenize input value로 입력.
    
  <img width="890" alt="Typed entity marker(punct)" src="https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/3c91cb47-194a-4a84-8bff-c1df75033266">

    
    - 결과 : roberta-small micro f1 기준: 56.2516→ 60.1862로 동일한 환경에서 유의미한 **성능 향상.**
- **Entity Mask**
    - 구현: sentence의 sbj, obj 위치에 [주체-sbj_type], [대상-obj_type] 대신 삽입
    - 결과: roberta-small train score 기준: 0.7237→ 0.6814로 동일한 환경에서 성능 하락**.**
- **Typed Entity Marker**
    - 구현: sentence의 sbj, obj 위치에 <주-sbj_type>, <대-obj_type> 대신 삽입
    - 결과: roberta-small train score 기준: 0.7237→ 0.7202로 동일한 환경에서 **성능 하락.**
- **UNK (언노운 토큰) Vocab 처리 [최종]**
    - 이유: Subject Entity와 Object Entity에 있는 단어가 Token화가 제대로 되지 않는 문제가 발생
    - 구현 : Tokenize 과정을 거친 후 UNK Token으로 바뀌는 entity를 Vocab에 추가. 230개 토큰 추가.
    - 결과 : 같은 조건으로 돌린 모델중 가장 우수한 성능을 보여줌.
- **한자를 한글로 번역 [최종]**
    - 이유: 한자를 한글로 번역하면 문장에 더 많은 정보가 담기지 않을까?
    - 구현: hanja 라이브러리를 사용해서, 한자를 한글로 번역함.
    - 결과: 한자의 경우 UNK Token으로 분류되는 단어가 한글로 번역하는 경우 UNK Token으로 변환되지 않아서, 성능을 향상시킬 수 있었음.
- **중복 문장 제거, 오라벨 데이터 제거**
    - 이유: 중복 문장, 오라벨 데이터들의 영향력을 줄여 학습 성능 향상 기대.
    - 구현 : pd.drop_duplicates(subset = id를 제외한 column),  id column을 제외하고 동일한 데이터 42개 제거. 조건 검색을 통해 id를 통한 데이터 제거
    - 결과 : 중복 문장을 제거하고, klue논문에 나와있는 라벨 설명과 맞지 않는 오라벨 데이터를 제거하여 학습 성능 향상을 기대하였지만, 성능 향상이 이루어지지 않아서 최종 모델에는 적용하지 않았다. test dataset에도 이러한 데이터가 존재한다면 성능 향상이 이루어지지 않을 수 있다고 생각한다.

### 4.4 Augmentation

- **Swap**
    - 분석 : subject entity와 Object entity를 바꿔도 label이 성립하는 경우가 있음. 혹은 특정 label로 교체가 가능한 경우를 확인. 이를 통해 데이터 증강이 가능할 것이라고 생각.
    - 결과 : 처음 가능한 모든 경우를 진행했을 때는 성능 하락. 이는 데이터에 비슷한 문장이 많고 sub, obj가 겹치는 경우가 많아서 성능이 하락한 것 같음. 최종 학습에는 이전에 잘 맞추지 못한 특정 라벨만 증강하여 사용 (일부만 증강하여 사용시 성능 향상).
- **LLM Augmentation(EleutherAI/polyglot-ko-1.3b, beomi/KoAlpaca-Polyglot-5.8B)**
    - LLM 프롬프팅을 통해 데이터 증강을 시도하였지만 원하는 품질의 학습 데이터를 얻지 못함.
- **BERT_Augmentation(K-TACC), Back translation**
    - K-TACC의 bert augmentation : subject entity와 object entity의 관계를 해치지 않기 위해 문장에 mask를 삽입하여 데이터 증강을 시도.
    - Back translation : 여러 언어로 역번역을 해보았을 때 sub, obj의 관계를 가장 해치지 않으면서 기존 문장과는 약간 다른 결과를 보였던 스페인어를 활용하여 증강을 시도.
    - 결과 : 두 방법 모두 기존 학습 데이터셋 내에 비슷한 문장이 많고 sub, obj가 겹치는 경우가 많아 학습 진행시 성능이 하락함.

### 4.5 Model

- **RoBERTa(large)**
    - 분석: 다수의 실험에서 RoBERTa-large 모델의 성능이 보편적으로 높은 것을 확인하여 Base pre-trained 모델로 사용하였다.
    - 결과: 성능 비교를 위한 실험과정에서는 RoBERTa-small을 사용해 전후 결과를 비교하였고, 최종 예측 단계에서는 RoBERTa-large 모델을 사용했다.
- **Task-adaptive pretraining(TAPT)[최종]**
    - 분석: Task-specific fine-tuning을 진행하기 앞서, 초기 initialization값으로 같은 task(RE)의 데이터셋을 이용해 Masked Language Modeling 사전학습 시킨 값을 사용하면 성능이 개선된다는 연구 결과
    - 결과: 대표 규칙에 따라 Test Dataset의 unlabeled 데이터 포함시킨 상태로 MLM 학습 진행. 규칙상 외부 RE task 데이터셋을 이용해 학습을 진행시키지는 않았기에 성능면에서 눈에 띄는 향상이 있지는 않았지만, 비슷한 정확도를 기록함에도 예측분포가 달라지는 것을 확인하였다.
- **no_realtion ↔ relation 이진분류**
    - 분석: EDA결과 분포의 30% 가량이 no_relation 라벨이고, baseline 실행 결과 오분류값에서도 no_relation의 비중이 크므로, no_relation을 제외한 나머지 29개의 라벨을 relation으로 통합하여 no_relation ↔ relation으로 이진분류 모델을 학습하고, no_relation을 제외한 나머지 29개를 분류하는 모델을 만들어 추론할 때 첫번째 모델에서 relation이라고 판단되면 두번째 모델로 추론해서 어떤 29개중 어떤relation인지 판단하였습니다.
    - 결과: training 과정에서 validation set에 대해서 최대 0.9998%의 정확도(7763/7764)를 기록했으나, 이를 리더보드에 제출해본 결과 성능 향상으로 이어지지 않았다.
        
  ![Binary classification1](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/4ba9b1ec-01a9-4e2c-8801-e501c308db7f)

        
- **no_relation → per:no_relation & org:no_relation 31진 분류**
    - 분석: train을 분석하다가 subject_entity의 ‘type’이 label의 앞 부분과 같은 것을 발견했고, label이 no_relation이 아닌 이상 ‘per:’ 또는 ‘org’로 시작하기 때문에 train에서 no_relation을 subject-entity의 type에 따라 ****per:no_relation & org:no_relation로 나누어 31개의 label로 학습을 진행하였다.
    - 결과: training 과정에서 validation set에 대해서 0.9958%의 정확도를 기록했으나, 이를 리더보드에 제출해본 결과 본래의 30개 예측보다 성능이 저하되었다.
- **Add biLSTM Layer**
    - 분석: 기존 모델이 no_relation label을 잘 구분하지 못하기 때문에 보다 복잡한 관계를 파악할 수 있게 Bert Layer와 FC layer 사이에 BiLSTM Layer를 추가
    - 결과: 기존 모델과 Inference 양상이 다르면서 성능 향상이 존재하여 앙상블 모델에 추가

### 4.6 Training & Hyperparameter Tuning

- **Imbalanced Data Sampler** [최종]
    - **기능** : 적은 데이터의 라벨에 큰 가중치를 부과하여 불균형 데이터셋에서 균형있게 샘플링
    - **구현** : 라벨을 5개의 구간으로 나누어 가중치 부과 (‘ImbalancedDatasetSampler’ 이용)
    - **결과** : 0.9187 → 0.9191로 **성능 향상**
- **klue/roberta-small 모델을 통한 성과 비교**
    - 작은 모델을 활용함으로써 빠르게 Ablation Study를 진행함
- **Optimizer 변경**
    - 기존 Optimizer는 AdamW를 사용하였으나, AdamP를 사용함.
    - AdamP는 Naver에서 구현한 최적화 모듈로 AdamW에 비해 우수하다고 알려졌으며, 실제 실험 결과 상으로도 성능이 좋은 것을 확인함.
- **Loss function 변경 [최종 → Label Smoothing Loss]**
    - 기존 Baseline model에서는 cross entropy를 loss function으로 사용
    - 데이터 불균형에 보다 적합한 Focal, Label Smoothing, LDAM Loss 등 사용
    - 실험 결과 **Label Smoothing Loss 사용시 성능 향상 있었음**
- **Max Length [최종 → 128]**
    - 문장 길이를 128, 160, 256으로 달리 지정하여 성능을 비교
- **Scheduler [최종 → CosineAnnealingLR]**
    - CosineAnnealingLR, OneCycleLR, CyclicLR 등 Scheduler를 비교 분석해서 성능이 가장 우수한 Scheduler를 사용함.
    - 실험 결과, **CosineAnnealingLR**의 성능이 가장 우수해서 해당 Scheduler를 사용함.
- **Batch Size [최종 → 64]**
    - Batch Size를 [16, 32, 64]로 설정하여, 가장 우수한 Batch Size를 선정함.
- **Learning Rate [최종 → 3e-5]**
    - Scheduler를 사용했기 때문에, Learning Rate는 5e-5, 3e-5 로 지정하여 최종적으로 3e-5를 선정함.

### 4.7 Evaluation & PostProcessing

  ![confusion_matrix](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/3e665de6-7c3b-4dd6-bc31-1d0900e94f05)


- **confusion matrix**, **정답 라벨과 예측 라벨을 비교**함으로써 모델의 취약점을 파악하고 개선하려함.

### 4.8 Ensemble

- **Soft Voting**
    - 구현 : output의 산술평균 계산(python 코드 작성)
    - 결과 : 대체적으로 Soft Voting을 수행할 경우 전체적인 성능이 향상
- **weight sum**
    - 구현 : ensemble을 위해 roberta-large, KoSimCSE, Electra 등의 모델 결과에 가중치를 부여해 가중합 방식으로 사용함.
    - 결과 :  단순히 ensemble하는 것보다, auprc가 전체적으로 향상됨. Roberta-large만 사용한 경우에는 micro-f1 score는 높게 나왔으나, auprc는 낮게 나오는 문제점을 완화함.

### 4.9 Dataset

| v0.0.1 | v0.0.2[최종] | v0.0.3 | v0.0.4 | v0.0.5[최종] |
| --- | --- | --- | --- | --- |
| torch.random_split(seed=0)을 이용해 0.8/0.2로 split | scikit-learn package를 사용해서 stratify=df[’label’] 지정 후 train/dev → 0.8/0.2 | v.0.0.2에서 train/dev → 0.9/0.1 | org:product 라벨 수정 | per:sibling sub↔obj swap |

### 4.10 Final Submission

- **실험 결과**

| Model Name | Batch Size | LR | Max Length | Scheduler | Dataset | micro f1 |
| --- | --- | --- | --- | --- | --- | --- |
| klue/bert-base | 16 | 1e-5 | 256 | x | v.0.0.0 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.0 | 61.4764 |
| monologg/koelectra-base-v3-discriminator | 16 | 5e-5 | 256 | x | v.0.0.0 | 28.7684 |
| BM-K/KoSimCSE-roberta | 128 | 5e-5 | 128 | x | v.0.0.1 | 63.6516 |
| snunlp/KR-ELECTRA-discriminator | 16 | 5e-5 | 128 | x | v.0.0.0 | 58.2706 |
| albert-base-v1 | 16 | 5e-5 |  | x | v.0.0.0 |  |
| kykim/electra-kor-base | 16 | 5e-5 | 256 | x | v.0.0.0 | 60.1513 |
| snunlp/KR-ELECTRA-discriminator | 64 | 5e-5 | 256 | x | v.0.0.0 | 58.3214 |
| klue/roberta-large | 16 | 5e-5 | 256 | x | v.0.0.2 | 62.1791 |
| klue/bert-base | 32 | 5e-5 | 256 | x | v.0.0.1 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 43.6494 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.0 |  |
| BM-K/KoSimCSE-roberta | 64 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 56.2516 |
| klue/roberta-large | 16 | 5e-5 | 256 | x | v.0.0.2 | 61.9115 |
| klue/roberta-large | 16 | 5e-5 | 256 | x | v.0.0.2 | 62.3393 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 47.9384 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 42.4432 |
| klue/roberta-large | 32 | 5e-5 | 256 | x | v.0.0.2 | 56.1932 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 60.1862 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 61.6119 |
| klue/roberta-large | 128 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 32 | 5e-5 | 256 | x | v.0.0.3 | 51.1585 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 6.6135 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.3 | 53.7552 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 61.5339 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 45.0580 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 44.2403 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 58.7462 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 57.8578 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 67.3778 |
| klue/roberta-small | 128 | 5e-5 | 128 | x | v.0.0.2 | 61.7949 |
| klue/roberta-small | 128 | 5e-5 | 128 | x | v.0.0.2 | 59.4228 |
| klue/roberta-large | 16 | 5e-5 | 256 | x | v.0.0.2 |  |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 60.5960 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 54.4690 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 70.2031 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 |  |
| BM-K/KoSimCSE-roberta | 128 | 5e-5 | 128 | x | v.0.0.2 | 66.0304 |
| klue/roberta-small | 128 | 5e-5 | 256 | x | v.0.0.2 | 62.6043 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 66.9538 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 70.0967 |
| klue/roberta-large,BM-K/KoSimCSE-roberta | 64 | 5e-5 | 128 | x | v.0.0.2 | 68.1204 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 68.8107 |
| BM-K/KoSimCSE-roberta | 64 | 5e-5 | 128 | x | v.0.0.2 |  |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.2 | 68.1777 |
| klue/roberta-large | 64 | 5e-5 | 128 | x | v.0.0.4 | 63.5180 |
| klue/roberta-large | 64 | 3e-5 | 160 | x | v.0.0.2 | 70.6522 |
| klue/roberta-large | 32 | 5e-5 | 128 | x | v.0.0.2 |  |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.4 | 71.0176 |
| klue/roberta-large | 32 | 3e-5 | 256 | x | v.0.0.2 | 72.0096 |
| klue/roberta-large | 32 | 3e-5 | 256 | x | v.0.0.2 | 71.9163 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.2 | 71.7348 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.2 | 72.0096 |
| klue/roberta-small | 64 | 5e-5 | 128 | x | v.0.0.2 | 61.3226 |
| kykim/electra-kor-base | 64 | 5e-5 | 128 | x | v.0.0.2 | 61.7457 |
| klue/roberta-large | 32 | 3e-5 | 256 | CosineAnnealingLR | v.0.0.2 | 71.4420 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR  |  | 73.0550 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR  |  | 74.4200 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.2 |  |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.5 | 73.4036 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.5 | 67.2598 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.5 | 72.6496 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.5 | 70.5786 |
| klue/roberta-large | 64 | 3e-5 | 128 | CosineAnnealingLR | v.0.0.5 |  |
- **Result**
    - Public LeaderBoard
  ![Public LeaderBoard](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/ee972513-053b-4e1c-aac6-cf3b9a42a9d5)

        
    - Private LeaderBoard     
  ![Private LeaderBoard](https://github.com/boostcampaitech6/level2-klue-nlp-01/assets/88371786/14025e17-bfa8-4d34-a684-8128d21d1765)

        

# 5. 자체 평가 의견

### 5.1 잘한 점

- Github, Notion 등 협업이 원활하게 이루어져서 실험 단계에서 다양한 요소를 실험해볼 수 있었던 점.
- 대회 초반부터 시간 분배를 잘하여 일정 계획한 점.
- 모델의 Inference 결과를 분석하여 다음 실험 계획을 설정한 점.
- 대회 초반부터 빠르게 코드를 모듈화해서, 하이퍼파라미터 수정으로 동일한 환경에서 실험 가능하게 세팅한 점.

### 5.2 시도 했으나 잘 되지 않았던 것

- LLM을 통한 증강을 시도했지만 프롬프트 엔지니어링 노하우의 부족으로 만족할만한 결과가 도출되지 않은 점.
- 데이터 증강 부분에서 좀 더 다양한 실험을 못해본 점.
- 모델 구조를 다양하게 변경해서 시도해보지 못한 점
- confidential learning 등 다양한 논문을 구현하고자 했으나, 성공하지 못한 점

### 5.3 아쉬웠던 점 → 개선 방안

- 프롬프트 엔지니어링에 대한 추가 학습을 통해 LLM 데이터 증강 시도
- 혼자서는 이해가 어려운 모델이나 다양한 training 기법에 대해, 팀원들이 같이 이해하고 역할 분담하기
- Max Length를 달리하면서 실험하는 과정에서 Subject Entity나 Object Entity 간의 관계를 제대로 파악할 수 있을 정도의 정보가 내포되어 있는지 파악하지 못했습니다.

### 5.4 프로젝트를 통해 배운 점 또는 시사점

- Max Length를 단순히 길이로 자르는 것이 아니라 다른 방식으로 자르는 것을 고안해낼 필요가 있습니다. 예를 들어, Subject Entity가 있는 문장을 가져온다는 식으로 처리를 할 수 있을 것 같습니다.
- 데이터 속의 메타데이터를 충분히 활용하는 것이 중요함을 배웠습니다. 모델의 취약점을 정확히 파악하기 위해선, 계획적인 EDA를 통해 데이터 자체에 대한 이해를 충분히 하는 것이 중요함을 배웠습니다.

## Reference

- Entity Mask, Typed Entity Marker, Typed entity marker(punct)
    
    https://arxiv.org/pdf/2102.01373.pdf
    
- K-TACC
https://github.com/kyle-bong/K-TACC
- KLUE-RE
    
    https://arxiv.org/pdf/2105.09680.pdf
    
- Label-Distribution-Aware Margin Loss
    
    https://arxiv.org/pdf/1906.07413.pdf
    
- Label Smoothing Loss
    
    https://arxiv.org/pdf/1906.02629.pdf
    
- TAPT
https://arxiv.org/pdf/2004.10964.pdf
