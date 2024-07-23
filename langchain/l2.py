import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

# Load the model and tokenizer from the local path
model_path = "/mnt/sda/agent_mxz/models/Qwen2-7B-Instruct"
# model_path = "/mnt/sda/agent_mxz/models/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
)

# Set up the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False,
    temperature=0,
    top_k=None,
    top_p=None,
    device='cuda',
    torch_dtype=torch.bfloat16
)
llm = HuggingFacePipeline(pipeline=pipe)

# Define a custom function to invoke the model
def invoke_model(prompt):
    response = llm.invoke(prompt)
    return response

# Example usage
prompt = "Hugging Face is"
response = invoke_model(prompt)
print(response)
