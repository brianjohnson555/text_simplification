# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import jieba
import pickle
import re

# vocab data:
blcu = pd.read_csv('../data/BLCU/literature_wordfreq.release_UTF-8.txt', header = None, sep="\t",)
blcu.rename(columns={0:"character", 1:"frequency"}, inplace=True)
blcu.set_index("character", inplace=True)
blcu["frequency"] = blcu["frequency"].rank(pct=True)
blcu = blcu.to_dict()['frequency']
with open("../data/HSK/HSK_levels.pickle", 'rb') as handle:
    hsk_dict = pickle.load(handle)

def extract_from_url(url):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # wait for JS-rendered content to load
    time.sleep(3)
    html = driver.page_source
    return BeautifulSoup(html, 'html.parser')

def parse_llm_output(response):
    # split into lines
    lines = response.strip().split('\n')

    # remove numeric prefixes
    re_lines = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if line.strip()]
    return re_lines

def build_simplification_prompt(sentences):
    prompt = ""
    base_instruction = '''请简化下面这段中文，使每个句子的用词更简单，适合中文学习者阅读。请特别注意以下几点：

1. 替换生僻词、高级词汇或抽象表达，使用更常见、更直白的词语或短语。
2. 避免使用超出HSK 6级范围的词汇，优先使用HSK 1-5级中常见词汇。
3. 保留原句数量和顺序，只进行词语层面的简化，不要省略、合并或总结内容。
4. 只输出简化后的句子，不要解释或分析。
5. 确保简化句子的语义与原句子保持一致。

原文段落：\n'''
    prompt += base_instruction

    for i, sentence in enumerate(sentences):
        if i<100:
            prompt+=f"{i}. {sentence}\n"
        else: break

    prompt+="\n请简化后的段落：\n"

    for ii in range(i):
        prompt+=f"{ii}. \n"
    
    return prompt

def build_judge_prompt(original, simplified):
    prompt = ""
    base_instruction = '''请对下列两个句子进行对比，判断简化句是否忠实保留了原句的语义，且语言通顺、自然。

如果简化句与原句意义相符且语言流畅，请直接输出简化句；
如果简化句表达不准确或不够自然，请你重新改写，使其既保留原句的主要信息，又更加容易被中文学习者理解。
\n'''
    prompt += base_instruction

    for i, sentence in enumerate(simplified):
        if i<100:
            prompt+=f"{i}. 原句：[{original[i]}], 简化句：[{sentence}]\n"
        else: break

    prompt+="\n请输出最终审核通过的句子：\n"
    for ii in range(i+1):
        prompt+=f"{ii}. \n"
    
    return prompt

def build_NER_prompt(simplified):
    prompt = ""
    base_instruction = '''请识别以下中文句子中的**专有名词类型的命名实体**，包括但不限于：

- 人名（具体的个人姓名）
- 地名（具体的国家、省市、街道等）
- 组织/机构名（公司、学校、政府机构等）
- 产品或书名、作品名

请不要标记普通名词（如“女儿”、“医院”、“保姆”）或泛指概念，只标记**具体的、唯一的、具有名称特征的实体**。

使用 HTML 的 `<b>` 标签将这些命名实体加粗，保留原句其他部分不变。

句子：
\n'''
    prompt += base_instruction

    for i, sentence in enumerate(simplified):
        if i<100:
            prompt+=f"{i}. {sentence}\n"
        else: break

    prompt+="\n请输出加粗后的句子：\n"
    for ii in range(i+1):
        prompt+=f"{ii}. \n"
    
    return prompt

def generate_html(url, sentences) -> str:
    """Returns an HTML-styled report based on job match output."""

    html = """
    <html>
    <head>
        <style>
            body {font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f5f5f5;}
            .job-card {background: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
            .job-title {font-size: 18px; font-weight: bold;}
            .company-location {font-size: 16px; color: #666;}
            .keywords {font-style: italic; margin-top: 10px;}
            .description {margin-top: 0; white-space: pre-wrap;}
            .description ul {margin-top: 0;}
            a {color: #1a0dab; text-decoration: none;}
            a:hover {text-decoration: underline;}
        </style>
    </head>
    <body>
    """
    html+=f"<h2>From {url}</h2>\n"

    for para in sentences:  # top 10 results
        html += f"<p>{para}</p>\n"

    html += "</body></html>"
    return html

def sentence_metrics(sentence):
    tokens = [word for word in jieba.cut(sentence)] # get tokens
    ## find portion of words in HSK level 1-3:
    levels = [hsk_dict[word] for word in tokens if word in hsk_dict]
    if levels:
        l13 = (levels.count(1) + levels.count(2) + levels.count(3))/len(tokens)
    else:
        l13 = 0
    ## find frequency of words:
    freqs = [np.power(blcu[word], 2) for word in tokens if word in blcu] # get squared frequency
    freq = np.mean(freqs) # mean of squared freqs

    return l13, freq

def corpus_metrics(complex_sentences: list, simple_sentences: list):
    simple_metrics = [sentence_metrics(sentence) for sentence in simple_sentences]
    complex_metrics = [sentence_metrics(sentence) for sentence in complex_sentences]
    l13_simple = np.mean([simple_metrics[idx][0] for idx in range(len(simple_metrics))])
    l13_complex = np.mean([complex_metrics[idx][0] for idx in range(len(complex_metrics))])
    freq_simple = np.mean([simple_metrics[idx][1] for idx in range(len(simple_metrics)) if not np.isnan(simple_metrics[idx][1])])
    freq_complex = np.mean([complex_metrics[idx][1] for idx in range(len(complex_metrics)) if not np.isnan(complex_metrics[idx][1])])
    l13_score = 100*(l13_simple - l13_complex)/l13_complex # percent change in L1-3 proportion
    freq_score = 100*(freq_simple - freq_complex)/freq_complex # percent change in squared frequency
    return l13_score, freq_score