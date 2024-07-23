from transformers import AutoModelForCausalLM, AutoTokenizer
import time
device = "cuda:0" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/sda/agent_mxz/models/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/sda/agent_mxz/models/Qwen1.5-14B-Chat")

prompt = "你好"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print('token: ', model_inputs.input_ids)
print(model_inputs.input_ids.shape)
print(type(model))
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=100
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)