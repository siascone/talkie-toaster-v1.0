# import os
# from dotenv import load_dotenv

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     raise ValueError("Hugging Face toke is missing. Please add it to the .env file")

from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login

MODEL_NAME = "distilgpt2"

# login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

