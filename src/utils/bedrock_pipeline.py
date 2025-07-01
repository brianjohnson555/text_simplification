import boto3
from botocore.exceptions import ClientError
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.chat_models import ChatBedrockConverse

# Create an Amazon Bedrock Runtime client.
brt = boto3.client("bedrock-runtime")

# Set the model ID
arn_llama = "arn:aws:bedrock:us-east-1:711387109035:inference-profile/us.meta.llama3-2-3b-instruct-v1:0"

llm_llama = ChatBedrockConverse(client=brt,
                        model_id=arn_llama,
                        provider="meta",
                        temperature=0.1,
                        max_tokens=150
                        ,)

def sentence_pipeline(sentence):
    return llm_llama.invoke(f"请简化下面的中文句子: {sentence}").content