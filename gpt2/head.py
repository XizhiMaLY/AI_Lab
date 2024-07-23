import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda"  # or "cpu" if you do not have a GPU

model_path = "/mnt/sda/agent_mxz/models/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

texts = ["Replace me by any text you'd like.", "Hello, this is"]
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Tokenize inputs
model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
print(model_inputs)

# Generate text with do_sample set to false
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

# Decode the generated text
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
