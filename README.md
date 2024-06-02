# WASSUP_FinalProject_team6

<img src = 'imgs/team_logo.png' width="200" height="200"/>

## EST soft WASSUP AI 개발과정 2기 파이널 프로젝트
**(Team 6)**
**팀원 및 담당 업무**
  + 김형범
  + 김현주
  + 최정민
  + **모든 인원이 모든 과정에 참여합니다. 담당 업무는 명목상 정했습니다. 모든 인원이 프로젝트를 이끌어 나가는 중 입니다**
---
## Our Subject
**실시간 감정분석 기반 LLM 챗봇**

주제 선정 동기, 주제 세부 사항 등 추후 추가

## Files
  + **image_crop.ipynb** : 이미지 crop 작업을 수행
  + **data_preprocess.py** : 모델 학습을 위해 데이터 전처리, 증강을 수행합니다.
  + **model.py** : 모델이 정의된 파일입니다.
  + **train.py** : train set을 사용해 모델을 학습하고 validation set으로 성능을 검증합니다.
  + **test.py** : 학습된 모델의 test set에 대한 성능을 측정합니다.
  + **demo.py** : 실시간 감정분석 챗봇을 실행합니다.


## How to use
  + 모델 학습

  image_crop.ipynb 를 이용해 각 데이터셋을 crop한 뒤 "./data/train/(감정명)/"에 저장합니다.
  
  이 과정을 train,validation,test set에 대해 진행합니다.
  
  원하는 train 옵션(batch_size, learning_rate, num_epochs)을 지정하여 train.py를 실행시켜 모델을 학습합니다.

  실행 예시
```
 $ python train.py --batch_size 64 --learning_rate 0.0005 --num_epochs 50
 ```
  + 모델 테스트
    학습된 모델은 아래 링크에서 다운받을 수 있습니다.

    https://drive.google.com/file/d/1flvY2gG-MUMKp5-aOVnxJsVfnv5BFaYt/view?usp=sharing

    학습된 모델 파일을 동일 경로에 저장한 후 다음 코드를 실행시켜 모델을 테스트합니다.
```
 $ python test.py
 ```
  + 실시간 감정분석 기반 LLM 챗봇 데모 실행
```
 $ python demo.py
 ```
