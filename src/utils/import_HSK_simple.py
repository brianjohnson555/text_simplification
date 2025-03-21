"""Reads HSK vocabulary files and compiles into a dictionary with word and HSK level data"""

### import packages
import pandas as pd
import numpy as np
import pickle

### data containing POS but not definition
HSK = pd.read_csv("./data/HSK/raw/hsk30.csv")
HSK.index = HSK['Simplified']
HSK.rename(columns={'Level':'level'}, inplace=True)
HSK.drop(labels=['Simplified', 'Traditional', 'Pinyin', 'WebNo', 'ID', 'POS', 'WebPinyin', 'OCR', 'Variants', 'CEDICT'], axis=1, inplace=True)
HSK.replace(to_replace='7-9', value='7', inplace=True)
HSK['level'] = HSK['level'].astype(int)

### save
# HSK_dict = HSK[HSK["level"]>3].to_dict()['level'] ### Only selecting HSK4 and above!
HSK_dict = HSK.to_dict()['level']
with open("./data/HSK/HSK_levels.pickle", "wb") as handle:
    pickle.dump(HSK_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)