---
title: "Fast Abstractive Summarization-RL 논문 리뷰"
date: 2021-11-18 14:06:12 -0400
categories: NLP
tag : Text-Summarization
use_math: true

---


# **Fast Abstractive Summarization-RL (Chen and Bansal et al., 2018, ACL)**

[📄**Paper : Fase Abstractive Summarization with Reinforce-Selected Sentence Rewriting**](https://aclanthology.org/P18-1063/)

> 강화 학습을 사용하여 Extractive Abstractive model을 연결한 end-to-end 프레임워크

<br>

**word-sentence hierarchical framework**   
→ Sentence level의 extract를 수행한 후, word-level의 rewrite 수행

<br>


## 💡 **Contribution 정리**

### 1.  sentence-level policy gradient method (RL)

<br>
sentence-level의 Extractor와 word-level의 Abstractor를 연결함으로써  

word-sentence hierarchy 프레임워크 구현

→ 언어 구조를 모델링하는 데 효과적이며, **병렬화 (_parallel decoding_)** 를 가능하게 함

<br>


### 2.  모델 속도 개선

<br>

extract와 rewrite이 병렬적으로 동작하는 parallel decoding로 인해 모델 속도 개선  
inference speed 10-20배 개선, training speed 4배 개선  

<br>
<br>

## 🌱 **세미나 자료**

<iframe src="/assets/files/Abstract-Meaning-Representation.pdf" width="100%" height="350px">
</iframe>