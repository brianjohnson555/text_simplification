# Text Simplification of Chinese language

## Description:
This repository is for a working project for Chinese text simplification. [See my project website](https://johnsonrobotics.com/Projects/textsimplification.html) for full details.

The goal of this project is to reduce the complexity of sentences in Chinese simplified script. While the primary audience is Chinese language learners looking to convert high-difficulty text into passages that are readable at a lower skill level, this work can also apply to simplifying Chinese content for Chinese readers with lower language vocabulary to improve media literacy.

The text simplification is achieved through two means:
1) a lexical simplification (LS) pipeline that uses NER, POS tagging, and cosine similarity to replace complex words with simpler synonyms.
2) a text simplification (TS) model that is implemented by fine-tuning a Chinese BART model on a Chinese wikipedia dataset.

The wikipedia dataset originates from Chong et al. [https://arxiv.org/abs/2306.02796], who machine-translated Chinese wikipedia to English, performed state-of-the-art text simplification in English, and machine-translated the simplified sentences back to Chinese to created a "pseudo" simplified sentence dataset.

The fine-tuned BART model is based on [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese). [My fine-tuned model is available here](https://huggingface.co/johnsonrobotics24/bart-base-chinese-textsimplification-v1.0).

## Usage:
Using the LS pipeline:
```
import src.utils.LS_pipeline as LS
simple_sentence = LS.LS_pipeline(sentence)
```

Loading the fine-tuned model from Huggingface for inference:
```
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("johnsonrobotics24/bart-base-chinese-textsimplification-v1.0")
model = BartForConditionalGeneration.from_pretrained("johnsonrobotics24/bart-base-chinese-textsimplification-v1.0")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

simple_sentence = text2text_generator(sentence, max_length=128, do_sample=False)[0]['generated_text'].replace(" ","")
```

The remaining files were used for preprocessing data, fine-tuning, and evaluating the results, and are not required for inferencing.

Author: Brian K. Johnson 2024
