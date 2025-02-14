"""Cross calculates cosine similarity of vocab from BLCU literature data. First 50,000 vocab words only."""
###### import statements
import pandas as pd
import numpy as np
import torch
import jieba.posseg as pseg
import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

#TODO: Fix dataframe.iloc statements for building similarity dict

###### read raw data
df = pd.read_csv('./data/BLCU/literature_wordfreq.release_UTF-8.txt', header = None, sep="\t",)
df.rename(columns={0:"character", 1:"frequency"}, inplace=True)
df.set_index("character", inplace=True)
df["frequency"] = df["frequency"].rank(pct=True)

###### calculate similarity of embeddings
vocab = list(df.iloc[:50000].index) # create vocab list
embeddings = model.encode(vocab) # create embeddings
similarity_t = model.similarity(vocab, vocab) # calculate similarity

###### sort by similarity value, get top 15 results
top_similar = torch.flip(np.argsort(similarity_t, axis=1)[:,-16:-1], dims=(1,)).numpy().tolist()

###### save
with open("./data/BLCU/top_similar.pickle", 'wb') as handle:
    pickle.dump(top_similar, handle, protocol=pickle.HIGHEST_PROTOCOL)

###### convert results to dict with matching POS tag V1: Using jieba
similarity_dict = {}
for idx in range(len(top_similar)):
    base_word = df.iloc[idx,0]
    top_list = list(df.iloc[top_similar[idx],0])
    flag = list(pseg.cut(base_word))[0].flag
    idx_list = []

    for word in top_list:
        tagged_word = list(pseg.cut(word))[0]  # POS tagging
        if tagged_word.flag==flag: 
            idx_list.append(word)
    if idx%500==0:
        print(idx)
    similarity_dict[base_word] = idx_list

###### convert results to dict with matching POS tag V2: using thulac
import thulac   
thu1 = thulac.thulac()

similarity_dict = {}
for idx in range(len(top_similar)):
    base_word = df.iloc[idx,0]
    top_list = list(df.iloc[top_similar[idx],0])
    flag = thu1.cut(base_word)[0][1]
    idx_list = []

    for word in top_list:
        tagged_word = thu1.cut(word)[0][1]  # POS tagging
        if tagged_word==flag: 
            idx_list.append(word)
    if idx%500==0:
        print(idx)
    similarity_dict[base_word] = idx_list

with open("./data/BLCU/similarity_dict.pickle", 'wb') as handle:
    pickle.dump(top_similar, handle, protocol=pickle.HIGHEST_PROTOCOL)
