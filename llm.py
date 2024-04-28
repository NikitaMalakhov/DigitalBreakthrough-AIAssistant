from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = torch.device('cuda')
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quantization_config)

model.config.use_cache = False

import warnings
warnings.filterwarnings("ignore")


from models import prompts
import importlib
importlib.reload(prompts)

from models import catboosty
import importlib
importlib.reload(catboosty)

from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=120)

def process(message: str, context: str):
    variants = catboosty.pipeline_predict2([message], catboosty.model_cl_category, catboosty.model_cl_answer, catboosty.tfidf_vectorizer)
    encodeds = tokenizer.encode(prompts.top_3_prompt(message, variants), return_tensors='pt')
    model_inputs = encodeds.to(device)
    gen_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    decoded = tokenizer.batch_decode(gen_ids)

    ans = decoded[0].split('[/INST]')[-1]

    encodeds = tokenizer.encode(prompts.get_prompt(message, ans, context), return_tensors="pt")

    model_inputs = encodeds.to(device)

    kwargs = dict(input_ids=model_inputs, streamer=streamer, max_new_tokens=500)

    thread = Thread(target=model.generate, kwargs=kwargs)

    thread.start()


import time
import gradio as gr
current_state = 'User'
messages = ""
loyalty = 100
def slow_echo(message, history):
    global messages, current_state
    messages += f"{current_state}: {message}\n"
    current_state = 'User' if current_state == 'Assistant' else 'Assistant'
    process(message, messages)
    history = ""
    for char in streamer:
        for k in char:
        history += char
        yield history

gr.ChatInterface(slow_echo, css="footer {visibility: hidden}").launch(share=True)