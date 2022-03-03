# CODE GUIDE

### Environment

```python
pip install -r requirements.txt
```

### Preprocess
차례로 노트북을 실행시켜 data폴더 아래로 데이터 생성 노트북 맨 위에 있는 split 변수를 train, val, test 순으로 변경하며 데이터 생성

data_process_rev.ipynb : specific query에 대한 data만 가져올 수 있음

make_sentence.ipynb : turn 단위로 relevant span이 들어가 있는지 없는지를 기준으로 labeling

### Train

get config에 있는 옵션을 통해 하이퍼 파라미터를 조절

ng_sentence_retriever : query와 relevant한 feature, negative feature 여러개로 구성된 feature로 dpr 학습하는 방법(아까 말씀드린 방법)

```python
python ng_sentence_retriever.py --lr 1e-5
```

in_batch_sentece_retriever : 단순히 query와 relevant span으로 dpr 학습

```python
python in_batch_sentence_retriever.py --lr 1e-5
```