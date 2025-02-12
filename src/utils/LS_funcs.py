"""Functions for lexical simplification (LS) task using HSK dictionary lookup and best candidate replacement."""

###### import packages, NER pipeline
import re
import jieba
import torch
import string
import pandas as pd
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-base-news')

###### Read HSK data
HSK = pd.read_pickle("../data/HSK/HSK_full")
top_choice = torch.tensor(HSK['top_choice']) # top replacement candidates (index)
top_choice_HSK = torch.tensor(HSK['top_choice_level']) # top replacement candidates (HSK level)

def replace_words(tokens, max_HSK):
    """Replaces input tokens with tokens at or below max_HSK level."""
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
    """Performs LS word replacement of input sentence at the given max_HSK level."""
    punc = ",!?！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.-" # possible punctuation
    sentence.translate(str.maketrans('', '', string.punctuation)) # convert str format
    text_re = re.sub(r"[%s]+" %punc, "", sentence) # remove punctuation

    tokens = jieba.lcut(text_re, cut_all=False) # tokenize
    tokens_l = list(dict.fromkeys(tokens)) #TODO: try: list(set(tokens))

    ner_output = ner_pipeline(text_re)['output'] # run NER pipeline, generate tokens
    tokens_ner = list(set([d['span'] for d in ner_output if len(d['span'])>1])) # only collect NER longer than 1 character
    tokens_no_ner = list(set(tokens_l) - set(tokens_ner)) # get all tokens that are NOT named entities

    replacement_dict = replace_words(tokens_no_ner, max_HSK) # replace all tokens except named entities
    sentence_new = sentence
    for key, value in replacement_dict.items():
        sentence_new = sentence_new.replace(key, value) # create output str

    return sentence_new

