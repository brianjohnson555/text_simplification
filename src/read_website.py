"""Takes the input url, retreives source html code, replaces words within
source, generates new html file with replaced words."""

###### import packages, sentence model, and parse html from site url
import requests
import re
from bs4 import BeautifulSoup
import jieba
import torch
import string
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
language_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

url = 'https://www.bbc.com/zhongwen/articles/c4gl97d2rzjo/simp'
site = requests.get(url)
site_soup = BeautifulSoup(site.text, 'html.parser')
HSK = pd.read_pickle("./data/Chinese/HSK_full")
# print(site_soup.prettify())

###### get text from the html and tokenize
site_p = site_soup.find_all('p')
text = ""
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏." # possible punctuation

for p in site_p:
    text += str(p.get_text()) # append new text to the text string

text.translate(str.maketrans('', '', string.punctuation)) # convert str format
text_re = re.sub(r"[%s]+" %punc, "", text) # remove punctuation marks

tokens = jieba.lcut(text_re) # tokenize
tokens_l = a = list(dict.fromkeys(tokens))

###### get text from the html and tokenize
site_s = site_soup.find_all(string=True)
text = ""
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏." # possible punctuation

for s in site_s:
    text += str(s.get_text()) # append new text to the text string

text.translate(str.maketrans('', '', string.punctuation)) # convert str format
text_re = re.sub(r"[%s]+" %punc, "", text) # remove punctuation marks

tokens = jieba.lcut(text_re) # tokenize
tokens_l = a = list(dict.fromkeys(tokens))

###### Cross compute the top embeddings for each vocab word and add to HSK dataframe
similarity_t = sentence_model.similarity(HSK["embedding"], HSK["embedding"])
top_choice = torch.flip(np.argsort(similarity_t, axis=1)[:,-21:-1], dims=(1,)).numpy().tolist()
top_choice_HSK = [[HSK["HSK"][i] for i in row] for row in top_choice] # convert from index values to HSK values

def simplify(tokens, max_HSK):
    simplified_tokens = dict()
    for token in tokens:
        try:
            char_idx = np.where(HSK["character"]==token)[0][0]
            if HSK["HSK"][char_idx]>max_HSK:
                top_idx = top_choice[char_idx][next(x[0] for x in enumerate(top_choice_HSK[char_idx]) if x[1] <= max_HSK)] # iterate through and find index of first element below max_HSK
                simplified_tokens[token] = HSK["character"].loc[top_idx]
            else:
                pass# simplified_tokens[idx] = 0
        except:
            pass# simplified_tokens[idx] = 1
    return simplified_tokens

replacement_dict = simplify(tokens_l, 4)

for element in site_soup.find_all(string=True):  # Get all text nodes
    text = element
    for old_word, new_word in replacement_dict.items():
        if old_word in text:
            text = text.replace(old_word, f'<span style="color: red;">{new_word}</span>')
    element.replace_with(BeautifulSoup(text, 'html.parser'))

with open("output.html", "w", encoding = 'utf-8') as file: 
    # prettify the soup object and convert it into a string 
    file.write(str(site_soup.prettify())) 