"""Pipeline for both text simplification (TS) with BART, and combined approach with LS_BART and BART_LS"""
import utils.LS_pipeline as LS
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("../data/models/batch_128_15625_500000", local_files_only=True)
model = BartForConditionalGeneration.from_pretrained("../data/models/batch_128_15625_500000", local_files_only=True)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

def TS_with_BART(sentence: str):
    simple_sentence = text2text_generator(sentence, max_length=128, do_sample=False)[0]['generated_text'].replace(" ","")
    return simple_sentence

def TS_with_BART_LS(sentence: str):
    bart_sentence = TS_with_BART(sentence)
    simple_sentence = LS.LS_pipeline(bart_sentence)
    return simple_sentence

def TS_with_LS_BART(sentence: str):
    ls_sentence = LS.LS_pipeline(sentence)
    simple_sentence = TS_with_BART(ls_sentence)
    return simple_sentence