'''
BERT Embedding
BERT input :
- [CLS] sentence [SEP]
- BERT vocab 안에 있는 token
- BERT tokenizer의 token ID
- mask ID  (뭐가 봐도 되는 토큰이고 아닌지) 1 or 0 : padding 0 , not padding 1
- segment ID (문장 구분) : input sentence가 한 문장이면 0, 두 문장일 때 : 첫문장은 0, 다음문장은 1
- positional embeddings (한 문장 내 특정 토큰의 순서)
'''

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from numpy.linalg import norm
def cosineSim(A,B):
    return np.dot(A,B) / (norm(A) * norm(B))

## Load pre-trained bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)

# Tokenization
# text = """Twitter put a 12-hour restriction on Donald Trump Jr.'s account,
# saying the president's son put out a tweet that contained misleading and potentially harmful information about the coronavirus."""
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."
print(text)

ClsSepToken = "[CLS] " + text + " [SEP]"
print(ClsSepToken)
tokenized_text = tokenizer.tokenize(ClsSepToken)
print(tokenized_text)
indexed_text = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text, indexed_text):
    print(tup[0], tup[1])


#segmentId
segment_id = [1] * len(tokenized_text)
print(segment_id)

token2Tensor = torch.tensor([indexed_text])
segments2Tensor = torch.tensor([segment_id])
##load weight of bert model
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
print(model)

with torch.no_grad():
    encoded_layer, _ = model(token2Tensor, segments2Tensor)
    print(len(encoded_layer)) ## num of layer
    print(len(encoded_layer[0])) ## num of batch
    print(len(encoded_layer[0][0])) ## num of token(word)
    print(len(encoded_layer[0][0][0])) # num of hidden units


    token_embedding = torch.stack(encoded_layer, dim=0)
    print(token_embedding.size()) ## [layer, batch, token, hidden units]

    token_embedding = torch.squeeze(token_embedding, dim=1)
    print(token_embedding.size()) ## [layer, toke, hidden_units]

    token_embedding = token_embedding.permute(1, 0, 2)
    print(token_embedding.size())
    
    
    ## 마지막 레이어 4개를 사용하는 방법이 성능이 좋다고 함
    ## concat, Sum 둘다 사용할 수 있음.
    token_cat = []
    token_sum = []
    for token in token_embedding:
        catTokenEmbedding = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_cat.append(catTokenEmbedding)
        sumTokenEmbedding = torch.sum(token[-4:], dim = 0)
        token_sum.append(sumTokenEmbedding)
    print(len(token_cat), len(token_cat[0]))
    print(len(token_sum), len(token_sum[0]))


    for idx, token in enumerate(tokenized_text):
        print(idx,"\t",token)

    ##print vector of token sum about bank
    print("Token Sum Vector about Bank)")
    print("bank", token_sum[6][:10])
    print("bank", token_sum[10][:10])
    print("river", token_sum[19][:10])

    ## print cosine simility
    print("Cosine simility")
    print("bank vs bank: ", cosineSim(token_sum[10],token_sum[6]))
    print("bank vs river: ", cosineSim(token_sum[10],token_sum[19]))
