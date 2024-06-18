#!/usr/bin/env python3

import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# os.environ['https_proxy'] = 'http://127.0.0.1:1081'

# MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'

sentences = ["Hello, world!"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
print(encoded_input)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
# normalize embeddings
# sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
formatter = {
    'float_kind': lambda x: f'{x:.5f}'
}
print("unnormalized embeddings:", np.array2string(sentence_embeddings.numpy(), threshold=1024, formatter=formatter))
