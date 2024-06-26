import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

device = "cuda:0"  # the device to load the model onto

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/home/agent_mxz/models/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/home/agent_mxz/models/Qwen1.5-14B-Chat")

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

# Create a dataset
class SimpleDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx]}

texts = [text]
dataset = SimpleDataset(texts)

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    model_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return model_inputs

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=1,
    predict_with_generate=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

# Predict with text generation enabled
predictions = trainer.predict(
    test_dataset=dataset,
)

# Decode the predictions
generated_ids = predictions.predictions
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("----------------------------------------")
print(response)
