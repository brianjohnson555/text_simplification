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
# print(site_soup.prettify())