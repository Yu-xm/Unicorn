import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import transformers
import pickle

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

import sys

# 把 /data/xmyu/Bunny_text/ 加进 sys.path，以便后续 import
sys.path.insert(0, "/data/xmyu/Bunny_text")
from bunny.model.language_model.bunny_llama import BunnyLlamaConfig, BunnyLlamaForCausalLM


# create model
model = AutoModelForCausalLM.from_pretrained(
    '/data/xmyu/Bunny_text/checkpoints-llama3-8b/bunny-llama3-8b/checkpoint-7364',
    torch_dtype=torch.float16, # float32 for cpu
    trust_remote_code=True).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    '/data/xmyu/Bunny_text/checkpoints-llama3-8b/bunny-llama3-8b/checkpoint-7364',
    trust_remote_code=True)

#  Is this anime style?

# text prompt
prompt = ""
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)


pkl_file = "image_features.pkl"
with open(pkl_file, "rb") as f:
    loaded_features = pickle.load(f)

image_features = torch.from_numpy(loaded_features).to("cuda")

# generate
output_ids = model.generate(
    input_ids,
    embeds=image_features,
    max_new_tokens=1024,
    use_cache=False,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
