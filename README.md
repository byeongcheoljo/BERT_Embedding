## BERT_Embedding 
### 구현 완료

## BERT Embedding 구현
#### Sentence : "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
#### the bank vault, the bank robber, the Mississippi river bank에서 bank는 은행이나 강둑(bank a river) 등을 의미할 수 있음
#### 첫번째와 두번째의 bank는 은행을 의미하고, 세번째의 bank는 강둑을 의미함  

### 각 단어에 대해 코사인 유사도를 구하면 다음과 같다:
### Cosine simility >>>
#### (the bank vault) bank vs (the bank robber) bank:  0.9456753
#### (the bank robber) bank vs the Mississippi river bank (bank):  0.67973334
