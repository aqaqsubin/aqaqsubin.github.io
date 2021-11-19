---
title: "MATCHSUM 실행 가이드"
date: 2021-07-25 17:33:18 -0400
categories: NLP
tag : Text-Summarization
---

# **MatchSum 실행 가이드**

### **📄Paper: MatchSum(Zhong et al., 2020, ACL)** 
Extractive Summarization as Text Matching


### **💻 Github Repo**  
https://github.com/maszhongming/MatchSum

<br>

|PAPER|SOURCE|TYPE|SAMPLE|TRAIN SIZE|VAILD SIZE|TEST SIZE|DOCUMENT TOKEN|SUMMARY TOKEN|
|---|---|---|---|---|---|---|---|---|
|Reddit|Social Media|SDS|[Reddit sample data](https://drive.google.com/file/d/1e_mxJdKTGuJQBkKb9-cvO84gp9bN36dG/view?usp=sharing)|41675|645|645|(average) 482.2|(average) 28.0|
|XSum|News|SDS|[XSum sample data](https://drive.google.com/file/d/1Fi3xrmzuh1zvG-WGksNs2VZ5IyjHqykl/view?usp=sharing)|203028|11273|11257|(average) 430.2|(average) 23.3|
|CNN/DailyMail|News|SDS|[CNNDM-RoBERTa sample data](https://drive.google.com/file/d/1ICSAjwX9YgU8XDqsWha1PA1XFpgFYnua/view?usp=sharing) <br>[CNNDM-BERT sample data](https://drive.google.com/file/d/1Tt6EeKqpc4kbtgRC3eCxRmYmLhj563QI/view?usp=sharing)|287084|13367|11489|(average) 766.1|(average) 58.2|
|WikiHow|Knowledge Base|SDS|[WikiHow sample data](https://drive.google.com/file/d/18fWe3pvGggjZbElfFp3knwyf8icOGKt2/view?usp=sharing)|168126|6000|6000|(average) 580.8|(average) 62.6 |
|PubMed|Scientific Paper|SDS|[PubMed sample data](https://drive.google.com/file/d/1wyLsXOMWg17hRBGwVqqvroF7SQNj1_hO/view?usp=sharing)|83233|4946|5025|(average) 444.0|(average) 209.5|
|Multi-News|News|MDS|[Multi-News sample data](https://drive.google.com/file/d/1SiOo0ffcrzgA02Vj3BTNBt7fEw5c10K6/view?usp=sharing)|44972|5622|5622|(average) 487.3|(average) 262.0|




> ***SDS*** : Single Document Summary  
> ***MDS*** : Multi Document Summary

<br>  

Zhong et al.의 연구에서는
CNN/DailyMail에 대해서, 2가지 버전의 전처리 데이터를 제공
(다른 데이터셋은 1개 버전)


```python
## Dependency

pytorch==1.4.0
fastNLP==0.5.0
pyrouge==0.1.3
rouge==1.0.0
transformers==2.5.1
```

  

### **Download Dataset**


1.  [CNN/DailyMail Dataset (BERT/RoBERTa ver)](https://drive.google.com/open?id=1FG4oiQ6rknIeL2WLtXD0GWyh6pBH9-hX)

	```python
	├── bert # BERT에 따라 전처리
	└── robert # RoBERTa에 따라 전처리
	```
  

2.  [Other (Reddit, XSum, WikiHow, PubMed, MultiNews)](https://drive.google.com/file/d/1PnFCwqSzAUr78uEcA_Q15yupZ5bTAQIb/view?usp=sharing)

	압축 해제한 `*.jsonl` 파일을 `MatchSum/data` 경로로 이동

<br>
    
    

###  **모델 학습**

  
**모델 저장 경로 설정**

```
export SAVEPATH=/<trained model save path>/
```

<br>

**모델 학습 파라미터 설정**

`gpus` 파라미터를 통해 사용 가능한 GPU 설정
`encoder` 파라미터를 통해 BERT 또는 RoBERTa 모델 선택 (*bert, roberta*)


Zhong et al.의 실험 환경은 8개의 *Tesla-V100-16G GPU*를 사용하였으며, 이에 학습은 30시간 소요

메모리에 따라 다음과 같이 조정하여 훈련할 수 있다.
  
-  `train_matching.py`의 `batch_size` 또는 `candidate_num`을 조정  
	`batch_size`  *(default=16)*
	`candidate_num`  *(default=20)*

-  `dataloader.py`의 `max_len` 값 지정  
	`class MatchSumLoader` 의 `max_len` (*default=180*)


```python
CUDA_VISIBLE_DEVICES=0,1 python train_matching.py --mode=train --encoder=roberta --save_path=$SAVEPATH --batch_size=8 --candidate_num=16 --gpus=0,1 
```
<br>
 
**모델 훈련 예시**

<br>  
<div align=left>
<img src="/assets/images/matchsum/matchsum_monitor.png" width=700/><br>
MATCHSUM(RoBERTa-base) 모델 훈련 
</div>
<br>

<br>

### **모델 검증**

학습이 끝나면 모델은 `$SAVEPATH` 내 모델의 학습 시작 시간 디렉토리 경로에 저장된다. (e.g. `/<trained model save path>/2020-04-12-09-24-51`)

<br>

**모델 경로 설정**

```
export MODELPATH=$SAVEPATH/<model training start time>
```

**모델 테스트**


📢 모델 테스트 시, GPU는 하나만 사용

```python
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=test --encoder=roberta --save_path=$MODELPATH --gpus=0
```

ROUGE 점수는 스크린에 나타나며, 학습된 모델은 `$SAVEPATH/result`에 저장된다
  
<br>

### **사전 학습된 모델**

- *CNN/DailyMail*  
	[MatchSum_cnndm_model.zip](https://drive.google.com/file/d/1PxMHpDSvP1OJfj1et4ToklevQzcPr-HQ/view?usp=drivesdk)

 
- *Other (MultiNews, PubMed, Reddit, WikiHow, XSum)*  
[ACL2020_other_model.zip](https://drive.google.com/file/d/1EzRE7aEsyBKCeXJHKSunaR89QoPhdij5/view?usp=drivesdk)

  

### **생성된 요약문 예시**

- *CNN/DailyMail*  
[ACL2020_output.zip](https://drive.google.com/file/d/11_eSZkuwtK4bJa_L3z2eblz4iwRXOLzU/view?usp=drivesdk)

  

- *Other (MultiNews, PubMed, Reddit, WikiHow, XSum)*  
[ACL2020_other_output.zip](https://drive.google.com/file/d/1iNY1hT_4ZFJZVeyyP1eeoVY14Ej7l9im/view?usp=drivesdk)