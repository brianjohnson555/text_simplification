"""Reads HSK vocabulary files from internet sources and compiles into single Pandas dataframe saved as pickle"""

### import packages
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # model for embedding

### data containing definition (english) but not POS (noun, verb, adj, etc.)
HSK_nums = [('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7_9', 7)] # HSK definitions to load files
HSK = pd.DataFrame()
for hsk in HSK_nums:
    hsk_df = pd.read_csv("../data/Chinese/HSK/HSK"+hsk[0]+".tsv", sep="\t", header=None)
    hsk_df.drop(0, axis=1, inplace=True)
    hsk_df.rename(columns={1: "character", 2: "pinyin", 3: "definition"}, inplace=True)
    hsk_df.insert(loc=2, column="HSK", value=[hsk[1]]*hsk_df.shape[0])
    hsk_df["embedding"] = hsk_df["definition"].map(sentence_model.encode) # create definition embedding
    HSK = pd.concat([HSK, hsk_df])
HSK.drop_duplicates("character", keep="first", inplace=True)
HSK.reset_index(inplace=True, drop=True)

### data containing POS but not definition
HSK2 = pd.read_csv("../data/Chinese/HSK/hsk30.csv")
HSK2.rename(columns={'Simplified':'character'}, inplace=True)
HSK2.drop(labels=['Traditional', 'Pinyin', 'WebNo', 'ID', 'WebPinyin', 'OCR', 'Variants', 'CEDICT','Level'], axis=1, inplace=True)

### merge
HSK_full = HSK2.merge(right=HSK, how='right', on='character')

### save
HSK_full.to_pickle("../data/Chinese/HSK_full")