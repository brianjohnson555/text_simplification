"""Pipeline for lexical simplification (LS) task using HSK dictionary lookup and best candidate replacement."""

###### import packages, NER pipeline
import re
import jieba
import torch
import string
import pandas as pd
import numpy as np
import jieba.posseg as pseg
import pickle
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-base-news')

from time import perf_counter

###### Load vocab data
with open("../data/BLCU/similarity_dict_v2.pickle", 'rb') as handle:
    similarity_dict = pickle.load(handle)
with open("../data/HSK/HSK_levels.pickle", 'rb') as handle:
    hsk_dict = pickle.load(handle)
# punc = ",!?！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.-" # possible punctuation
blcu = pd.read_csv('../data/BLCU/literature_wordfreq.release_UTF-8.txt', header = None, sep="\t",)
blcu.rename(columns={0:"character", 1:"frequency"}, inplace=True)
blcu.set_index("character", inplace=True)
blcu["frequency"] = blcu["frequency"].rank(pct=True)

def LS_pipeline(sentence: str, verbose:bool = False):
    """End-to-end pipeline for lexical simplification of a sentence."""
    tstart = perf_counter()
    ### tokenize sentence:
    tokens = jieba.lcut(sentence)
    tokens = [token for token in tokens if not re.match(r'^\W+$', token)] # remove punctuation

    ### NER on each token:
    tner = perf_counter()
    ner_output = ner_pipeline(sentence)['output'] # run NER pipeline, generate tokens
    print("ner time is:", perf_counter()-tner)
    tokens_ner = list(set([d['span'] for d in ner_output if len(d['span'])>1])) # only collect NER longer than 1 character
    tokens_no_ner = list(set(tokens) - set(tokens_ner)) # get all tokens that are NOT named entities
    
    ### get complex words:
    tcomp = perf_counter()
    complex_words = find_complex_words(tokens_no_ner)
    print("complex_word time is:", perf_counter()-tcomp)

    ### get candidates and build new sentences:
    simple_sentence = sentence
    for word in complex_words:
        try:
            candidates = similarity_dict[word]
            # remove candidates with lower frequency:
            word_freq = blcu.loc[word].values[0]
            for candidate in candidates:
                cand_freq = blcu.loc[candidate].values[0]
                if cand_freq < word_freq*0.99:
                    candidates.remove(candidate)

            ### select final candidate, replace in sentence:
            if candidates:
                tchoose = perf_counter()
                simple_sentence = choose_and_replace(simple_sentence, word, candidates) # update new sentence with replaced words
                print("choose time is:", perf_counter()-tchoose)
        except:
            pass
        
    if verbose:
        print("Original sentence:\n", sentence)
        print("Simplified sentence:\n", simple_sentence)
        print("NER:\n", tokens_ner)
        for word in complex_words:
            print("Complex word: ", word)
            if word in similarity_dict:
                print("Candidates: ", [f"{cand}" for cand in similarity_dict[word]])
            else:
                print("Outside of dictionary")
    print("pipeline time is:", perf_counter()-tstart)
    return simple_sentence

def find_complex_words(tokens: list, HSK_thresh:int = 5, freq_thresh:float = 0.98):
    """Finds complex words based on one of two criteria: HSK level, and frequency in BLCU data. Assumes
    less frequent words are more difficult or less likely to be known."""
    complex_HSK = [token for token in tokens if (token in hsk_dict and hsk_dict[token]>HSK_thresh)] # find all tokens above HSK level
    simple_HSK = [token for token in tokens if (token in hsk_dict and hsk_dict[token]<=HSK_thresh)] # also find simple tokens
    complex_freq = [token for token in tokens if (token in blcu.index and blcu.loc[token].values[0]<freq_thresh)] # find all tokens below freq level
    complex_words = list(set(complex_HSK).union(set(complex_freq).difference(set(simple_HSK)))) # combine by union
    return complex_words

def choose_and_replace(sentence: str, word: str, candidates: list):
    print("\t Entering choose func...")
    embed_orig = model.encode(sentence)
    similarity = []
    for candidate in candidates:
        new_sentence = sentence.replace(word, candidate) # new sentence with word replaced
        tstart = perf_counter()
        embed_new = model.encode(new_sentence) # create embeddings
        print("cand embed time:", perf_counter()-tstart)
        tstart2 = perf_counter()
        similarity.append(np.float32(model.similarity(embed_orig, embed_new)[0][0])) # calculate similarity
        print("cos sim time:", perf_counter()-tstart2)
    best_idx = int(np.argmax(similarity))
    sentence = sentence.replace(word, candidates[best_idx])

    return sentence