'''
BERT Embedding
BERT input :
- [CLS], [SEP]
- BERT vocab 안에 있는 token
- BERT tokenizer의 token ID
- mask ID  1 or 0 : padding 0 , not padding 1
- segment ID (문장 구분) : input sentence가 한 문장이면 0, 두 문장일 때 : 첫문장은 0, 다음문장은 1
- positional embeddings (한 문장 내 특정 토큰의 순서)

'''

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
## Load pre-trained bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization
text = """Twitter put a 12-hour restriction on Donald Trump Jr.'s account,
saying the president's son put out a tweet that contained
misleading and potentially harmful information about the coronavirus"""
print(text)
