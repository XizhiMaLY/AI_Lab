import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model_path = "/mnt/sda/agent_mxz/models/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

texts = ["Replace me by any text you'd like.", "Hello, this is"]
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
print(model_inputs)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

print(tokenizer.batch_decode(generated_ids)[1])