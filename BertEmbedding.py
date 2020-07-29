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
## Load pre-trained bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = False)

# Tokenization
text = """Twitter put a 12-hour restriction on Donald Trump Jr.'s account,
saying the president's son put out a tweet that contained misleading and potentially harmful information about the coronavirus."""
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
