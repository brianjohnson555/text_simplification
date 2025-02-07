"""Reads HSK vocabulary files from internet sources and compiles into single Pandas dataframe saved as pickle"""

### import packages
import pandas as pd
import numpy as np
import pickle

### data containing definition (english) but not POS (noun, verb, adj, etc.)
HSK_nums = [('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7-9', 7)] # HSK definitions to load files
HSK = pd.DataFrame()

### data containing POS but not definition
HSK = pd.read_csv("./data/Chinese/HSK/hsk30.csv")
HSK.index = HSK['Simplified']
HSK.rename(columns={'Level':'level'}, inplace=True)
HSK.drop(labels=['Simplified', 'Traditional', 'Pinyin', 'WebNo', 'ID', 'POS', 'WebPinyin', 'OCR', 'Variants', 'CEDICT'], axis=1, inplace=True)
HSK.replace(to_replace='7-9', value='7', inplace=True)
HSK['level'] = HSK['level'].astype(int)

### save
HSK_dict = HSK[HSK["level"]>3].to_dict()['level'] ### Only selecting HSK4 and above!
with open("./data/Chinese/HSK_levels.pickle", "wb") as handle:
    pickle.dump(HSK_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# HSK.to_pickle("./data/Chinese/HSK_levels")