---
title: "GSum 실행 가이드"
date: 2021-11-19 10:14:37 -0400
categories: NLP
tag : Text-Summarization
use_math: true

---

<br>

# **GSum 실행 가이드**

### **📄Paper: GSum (Dou et al., 2021, NAACL)** 
GSum: A General Framework for Guided Neural Abstractive Summarization

### **💻 Github Repo**  
https://github.com/neulab/guided_summarization


<br>

| Dataset | Source | Type | Train size | Valid size | Test size | Document Token | Summary Token |
|--|--|--|--|--|--|--|--|
| Reddit (TIFU-long) | Social Media | SDS | 41675 | 645 | 645 | (avg.) 482.2 |  (avg.) 28.0 |
| XSum | News | SDS | 203028 | 11273 | 11257 | (avg.) 430.2 |  (avg.) 23.3 |
| CNN/DailyMail (Non_Anonymized) | News | SDS | 287084 | 13367 | 11489 | (avg.) 766.1 |  (avg.) 58.2 |
| WikiHow | Knowledge Base | SDS | 168126 | 6000 | 6000 | (avg.) 580.8 |  (avg.) 62.6 |
| New York Times | News | SDS |  |  |  |  |   |
| PubMed | Scientific Paper | SDS | 83233 | 4946 | 5025 | (avg.) 444.0 |  (avg.) 209.5 |


> 📢 **SDS** : Single Document Summary  

NYT 데이터셋은 License가 필요하기 때문에 정확한 정보를 파악하지 못했다.  

<br>

```python
## Dependency

multiprocess==0.70.9
numpy==1.17.2
pyrouge==0.1.3
pytorch-transformers==1.2.0
tensorboardX==1.9
torch==1.1.0
fairseq==0.10.2
rouge==1.5.5
```

📢 Dou et al.의 연구에서는 **BertAbs** (Liu and Lapata, 2019)와 **BART** (Lewis et al., 2020) 모델을 사용하여성능을 측정함

<br>

---

## **BertAbs 기반 Guidance Summarization**
<br>

📣 현재 경로 `<home>/guided_summarization/bert`

<br>

- ### **Download Preprocessed Dataset**
    
    Liu and Lapata의 데이터 전처리를 그대로 사용하기 때문에, [***PreSumm repository***](https://github.com/nlpyang/PreSumm)에서 배포한 데이터를 사용함
    
    [bert_data_cnndm_final.zip](https://drive.google.com/file/d/1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI/view?usp=drivesdk)
    
- ### **모델 학습**
    
    경로 설정  
    ```bash
    export DATA_DIR_PATH=/<processed data directory path>/
    export MODEL_DIR_PATH=/<model directory path>/
    export LOG_PATH=/<training log path>/training.log
    export DATA_PATH=$DATA_DIR_PATH/cnndm
    ```

    <br>

    > `$DATA_DIR_PATH` 내 데이터들은 `cnndm.train.1.bert.pt`의 형식으로 저장되어 있음
    > 
    
    <br>

    📢 BERT 모델을 다운로드하기 위해, 첫번째 실행일 경우 아래 커맨드의 `visible_gpus` 를 -1로 입력한다.  
    모델 다운로드가 끝나면 종료한 후, 환경에 맞게 multi-GPU를 설정할 수 있다.
    
   
    
    ```bash
    python z_train.py -task abs -mode train \
    				-bert_data_path $DATA_PATH \
    				-dec_dropout 0.2 \
    				-model_path $MODEL_PATH \
    				-sep_optim true \
    				-lr_bert 0.002 \
    				-lr_dec 0.2 \
    				-save_checkpoint_steps 2000 \
    				-batch_size 140 \
    				-train_steps 200000 \
    				-report_every 50 \
    				-accum_count 5 \
    				-use_bert_emb true \
    				-use_interval true \
    				-warmup_steps_bert 20000 \
    				-warmup_steps_dec 10000 \
    				-max_pos 512 \
    				-visible_gpus 0,1 \ # -1 for first run 
    				-log_file $LOG_PATH
    ```
    
    <br>

- ### **모델 테스트**
    
    
    생성된 요약문 저장 경로 설정
    
    ```bash
    export RESULT_PATH=/<generated summaries path>/
    ```
    
    
    📢 모델 테스트 시, GPU는 하나만 사용
    
    
    ```bash
    python -W ignore z_train.py \
        -task abs \
        -mode test \
        -batch_size 3000 \
        -test_batch_size 1500 \
        -bert_data_path $DATA_PATH \
        -log_file logs/test.logs \
        -test_from $MODEL_PATH \
        -sep_optim true \
        -use_interval true \
        -visible_gpus 0 \
        -max_pos 512 \
        -max_length 200 \
        -alpha 0.95 \
        -min_length 50 \
        -result_path $RESULT_PATH
    ```
    
<br>

---

## **BART 기반 Guidance Summarization**
<br>

📣 현재 경로 `<home>`

<br>

- ### **데이터 전처리**
    <br>
    
    **Download Source Dataset and Guidance**
    
    BART 모델을 위한 Non-Anonymized CNN/DailyMail 데이터셋을 사용하기 위해
     [***sensim repository***](https://github.com/icml-2020-nlp/semsim)에서 배포한 데이터를 사용함
    
    [cnn/dailymail dataset zipfile](https://drive.google.com/file/d/1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by/view?usp=drivesdk)  
    [guidance zipfile for cnn/dailymail](https://drive.google.com/file/d/12SpWwfD3syIxcC-SdSNnDOI5sbXJaylC/view?usp=drivesdk)
    
    <br>

    **Encode with GPT2-BPE**
    
    
    
    ```bash
    git clone https://github.com/pytorch/fairseq.git  
    cd fairseq # 현재 경로 <home>/fairseq

    wget -N https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json  
    wget -N https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    wget -N https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
    ```

    ```bash
    export DATA_DIR_PATH=/<path to downloaded data>/cnn_dm
    export GUIDANCE_DIR_PATH=/<path to downloaded guidance>
    ```

    `$DATA_DIR_PATH` 경로 내 `train.source`, `train.target`, `val.source`, `val.target` 데이터에 Byte-pair Encoding
    
    ```bash
    for SPLIT in train val
    	do
    		for LANG in source target
    		do
    			python -m examples.roberta.multiprocessing_bpe_encoder \
    			--encoder-json encoder.json \
    			--vocab-bpe vocab.bpe \
    			--inputs $DATA_DIR_PATH/$SPLIT.$LANG \
    			--outputs $DATA_DIR_PATH/$SPLIT.bpe.$LANG \
    			--workers 60 \
    			--keep-empty
    	done
    done
    ```
    
    `$GUIDANCE_DIR_PATH` 경로 내 `train.oracle`, `val.oracle` 데이터에 Byte-pair Encoding
    
    ```bash
    for SPLIT in train val
    	do 
    	python -m examples.roberta.multiprocessing_bpe_encoder \
    	--encoder-json encoder.json \
    	--vocab-bpe vocab.bpe \
    	--inputs $GUIDANCE_DIR_PATH/$SPLIT.oracle \
    	--outputs $DATA_DIR_PATH/$SPLIT.bpe.z \
    	--workers 60 \
    	--keep-empty
    done
    ```
    
    **Binarize Dataset**
    
    `guided_summarization` 디렉토리로 이동 (현재 경로 `<home>/guided_summarization` )
    
    ```bash
    wget -N https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
    ```

    데이터 저장 경로 설정
    
    ```bash
    export BIN_DIR_PATH=/<output binarized data directory path>/
    ```

    ```bash
    python fairseq_cli/guided_preprocess.py \
      --source-lang source \
      --target-lang target \
      --trainpref **$DATA_DIR_PATH**/train.bpe \
      --validpref **$DATA_DIR_PATH**/val.bpe \
      --destdir **$BIN_DIR_PATH** \
      --workers 60 \
      --srcdict dict.txt \
      --tgtdict dict.txt
    ```
    
    <br>

- **모델 학습**
    
    
    학습 파라미터 설정

    ```bash
    export MODEL_PATH=/<model save path>/
    export LOG_PATH=/<training log directory path>/training.log
    export BART_PATH=<checkpoint filename>
    ```

    (학습된 모델은 `$MODEL_PATH/$BART_PATH`의 경로에 저장됨)
    
    ```bash
    export TOTAL_NUM_UPDATES=20000
    export WARMUP_UPDATES=500
    export LR=3e-05
    export MAX_TOKENS=2048
    export UPDATE_FREQ=16
    ```

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python train.py $BIN_DIR_PATH\
        --restore-file $BART_PATH \
        --max-tokens $MAX_TOKENS \
        --task guided_translation \
        --source-lang source --target-lang target \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --arch guided_bart_large \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --save-dir $MODEL_PATH \
        --find-unused-parameters
    ```
    
    - 학습된 모델    
        [bart_sentence.pt](https://drive.google.com/file/d/1BMKhAh2tG5p8THxugZWMPc7NXqwJDHLw/view?usp=drivesdk)
        
    <br>

- **모델 테스트**
    
    
    **테스트 파라미터 설정**
    
    ```bash
    export TEST_DATA_PATH=/<test source data path>/
    export TEST_GUIDANCE=/<test guidance path>/
    export RESULT_PATH=/<generated summaries path>/
    export MODEL_DIR=/<trained model path>/
    ```

    *Optional Parameter*
    
    ```bash
    export MODEL_NAME=/<trained model name>/ # default : model.pt  
    export DATA_BIN=/<trained data path>/ # default : .
    ```

    <br>

    **요약문 생성**
    
    ```bash
    python z_test.py $TEST_DATA_PATH $TEST_GUIDANCE $RESULT_PATH $MODEL_DIR $MODEL_NAME $DATA_BIN
    ```
    <br>

    **결과 파일로부터 ROUGE score 계산**
    
    1) **Files2ROUGE 다운로드**

    ```bash
    # Install Requirement
    pip install -U git+https://github.com/pltrdy/pyrouge
    ```
        

    ```bash
    # Clone the repo, setup the module and ROUGE
    git clone https://github.com/pltrdy/files2rouge.git
    cd files2rouge
        
    python setup_rouge.py
    python setup.py install
    ```
    
    2) **Stanford CoreNLP 4.2.2 (for tokenize) 다운로드**  
    
    [Stanford CoreNLP 4.2.2 다운로드](https://stanfordnlp.github.io/CoreNLP/)
    
    
    ```bash
    export CLASSPATH=/<corenlp_download_path>/stanford-corenlp-4.2.2.jar
    ```
    
    3) **ROUGE score 계산**   
    📢 ROUGE-1.5.5 버전을 설치해야함  
        
    ```bash
    cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
    cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target

    files2rouge test.hypo.tokenized test.hypo.target
    ```
