from evaluate import load
import numpy as np
import jieba
import pandas as pd
import pickle

# vocab data:
blcu = pd.read_csv('../data/BLCU/literature_wordfreq.release_UTF-8.txt', header = None, sep="\t",)
blcu.rename(columns={0:"character", 1:"frequency"}, inplace=True)
blcu.set_index("character", inplace=True)
blcu["frequency"] = blcu["frequency"].rank(pct=True)
blcu = blcu.to_dict()['frequency']
with open("../data/HSK/HSK_levels.pickle", 'rb') as handle:
    hsk_dict = pickle.load(handle)

# MCTS sentence data
with open('../data/mcts/mcts.dev.orig', encoding="utf8") as f:
    mcts = f.readlines()
mcts_ref = []
for dataset in range(0,5):
    filename = str('../data/mcts/mcts.dev.simp.'+str(dataset))
    with open(filename, encoding="utf8") as f:
        mcts_ref.append(f.readlines())

# scoring metrics:
sari = load("sari")
bertscore = load("bertscore")
bleu = load("bleu")

def chinese_tokenizer(text):
    return " ".join(jieba.cut(text))

def saribleu_metrics(predictions, references, sources):
    predictions_tokenized = [chinese_tokenizer(s) for s in predictions]
    references_tokenized = [[chinese_tokenizer(s)] for s in references]
    # references_tokenized = [[chinese_tokenizer(s) for s in references[i]] for i in range(5)]
    sources_tokenized = [chinese_tokenizer(s) for s in sources]
    results_sari = sari.compute(sources=sources_tokenized, predictions=predictions_tokenized, references=references_tokenized)
    results_bleu = bleu.compute(predictions=predictions_tokenized, references=references_tokenized)
    results_bert = bertscore.compute(predictions=predictions_tokenized, references=references_tokenized, model_type="distilbert-base-uncased")
    return results_sari["sari"], results_bleu["bleu"], np.mean(results_bert["f1"])

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

def corpus_metrics(sources: list, predictions: list):
    simple_metrics = [sentence_metrics(sentence) for sentence in sources]
    complex_metrics = [sentence_metrics(sentence) for sentence in predictions]
    l13_simple = np.mean([simple_metrics[idx][0] for idx in range(len(simple_metrics))])
    l13_complex = np.mean([complex_metrics[idx][0] for idx in range(len(complex_metrics))])
    freq_simple = np.mean([simple_metrics[idx][1] for idx in range(len(simple_metrics)) if not np.isnan(simple_metrics[idx][1])])
    freq_complex = np.mean([complex_metrics[idx][1] for idx in range(len(complex_metrics)) if not np.isnan(complex_metrics[idx][1])])
    l13_score = 100*(l13_simple - l13_complex)/l13_complex # percent change in L1-3 proportion
    freq_score = 100*(freq_simple - freq_complex)/freq_complex # percent change in squared frequency
    return l13_score, freq_score