
from transformers import AutoTokenizer, Qwen2ForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# print(os.environ['sdpa']=='False')
import torch

# Model and tokenizer paths
model_path = "/mnt/sda/agent_mxz/models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen2ForCausalLM.from_pretrained(model_path)

# Input texts
texts = ["Replace me by any text you'd like.", "Hello, this is", "Write a story for me.", "I love", "你", "12345"]

# Ensure padding is done on the left
tokenizer.padding_side = "left"

# Define PAD Token = EOS Token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.config.do_sample = False


# Tokenize the inputs with padding
encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
print(encoded_inputs)
# Get model output
# outputs = model(**encoded_inputs)
outputs1 = model.generate(**encoded_inputs, max_new_tokens=100, do_sample=False)
# print('outputs1', outputs1)
# print(tokenizer.batch_decode(outputs1)[1])
# Print the outputs
# print(outputs[0][1])
# print(tokenizer.batch_decode(torch.argmax(outputs[0], dim=-1)))