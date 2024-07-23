from transformers import GPT2Tokenizer, GPT2Model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

# Model and tokenizer paths
model_path = "/mnt/sda/agent_mxz/models/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2Model.from_pretrained(model_path)

# Input texts
texts = ["Replace me by any text you'd like.", "Hello, this is", "Write a story for me."]

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
outputs = model(**encoded_inputs)

# Print the outputs
# print(outputs[0][1])
print(tokenizer.batch_decode(torch.argmax(outputs[0], dim=-1)))