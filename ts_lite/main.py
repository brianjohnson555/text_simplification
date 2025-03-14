from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the model
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("./ts_lite/model", local_files_only=True)
model = BartForConditionalGeneration.from_pretrained("./ts_lite/model", local_files_only=True)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

# Create BaseModel class for app
class Input(BaseModel):
    sentence: str

@app.post("/predict/")
async def predict(input: Input):
    simple_sentence = text2text_generator(input.sentence, max_length=128, do_sample=False)[0]['generated_text'].replace(" ","")
    return {"simplified": simple_sentence}

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)