from transformers import LlavaForConditionalGeneration

from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("/mnt/sda/agent_mxz/models/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("/mnt/sda/agent_mxz/models/llava-1.5-7b-hf")

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=30)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
