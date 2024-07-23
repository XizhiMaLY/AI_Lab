from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

device = "cuda"  # the device to load the model onto
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_path = "/mnt/sda/agent_mxz/models/Baichuan2-7B-Chat"
model_path = "/mnt/sda/agent_mxz/models/Qwen2-7B-Instruct"
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
import time
# print("sleep")
# time.sleep(8)
# Define the prompt and messages
prompt = "你是一个数据科学家，这里有一个表结构，表ioms_alarm_current 字段：alarm_id, line, system, device_id, level, station, sub_frame, slot, port, alias, code, content, target_id, abnormal_type, trigger_reason, count, alarm_time_first, alarm_time_last, confirm_time, status, source, alarm_type, remark, 查询一下最近三天的告警信息。要求你仅输出SQL代码,不加入任何非代码以外的解释或者背景信息，不要解释，不要解释，不要解释"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply the chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Define the generation configuration with adjusted temperature
generation_config = GenerationConfig(
    temperature=0.001,  # Adjust the temperature as needed
    max_new_tokens=200,
    do_sample=False
)

# Generate the response
with torch.no_grad():
    generated_ids = model.generate(
        model_inputs.input_ids,
        generation_config=generation_config
    )

# Extract the generated tokens
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the response
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

