###### import packages, sentence model, and parse html from site url
import requests
import re
import jieba
import torch
import string
import pandas as pd
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-base-news')

HSK = pd.read_pickle("../data/Chinese/HSK_full")
top_choice = torch.tensor(HSK['top_choice'])
top_choice_HSK = torch.tensor(HSK['top_choice_level'])

def replace_words(tokens, max_HSK):
    simplified_tokens = dict()
    for token in tokens:
        try:
            char_idx = np.where(HSK["character"]==token)[0][0]
            if HSK["HSK"][char_idx]>max_HSK:
                top_idx = top_choice[char_idx][next(x[0] for x in enumerate(top_choice_HSK[char_idx]) if x[1] <= max_HSK)] # iterate through and find index of first element below max_HSK
                simplified_tokens[token] = HSK["character"].loc[int(top_idx)]
            else:
                pass# simplified_tokens[idx] = 0
        except:
            pass# simplified_tokens[idx] = 1
    return simplified_tokens

def simplify(sentence: str, max_HSK: int):
    punc = ",!?！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.-" # possible punctuation
    sentence.translate(str.maketrans('', '', string.punctuation)) # convert str format
    text_re = re.sub(r"[%s]+" %punc, "", sentence) # remove punctuation marks

    tokens = jieba.lcut(text_re, cut_all=False) # tokenize
    tokens_l = list(dict.fromkeys(tokens)) #TODO: try: list(set(tokens))

    ner_output = ner_pipeline(text_re)['output']
    tokens_ner = list(set([d['span'] for d in ner_output if len(d['span'])>1])) # only collect NER longer than 1 character
    tokens_no_ner = list(set(tokens_l) - set(tokens_ner))

    replacement_dict = replace_words(tokens_no_ner, max_HSK) # do not replace NER

    sentence_new = sentence
    for key, value in replacement_dict.items():
        sentence_new = sentence_new.replace(key, value)

