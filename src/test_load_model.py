# import os
# from dotenv import load_dotenv

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     raise ValueError("Hugging Face toke is missing. Please add it to the .env file")

from transformers import pipeline
from load_model import model, tokenizer
# from huggingface_hub import login

# login(token=HF_TOKEN)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Hello, how are you today?"
response = chatbot(
    prompt, 
    max_length=50, 
    truncation=True
)

print("Chatbot Response:", response[0]['generated_text'])