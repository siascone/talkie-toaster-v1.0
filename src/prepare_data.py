from transformers import GPT2TokenizerFast
from datasets import Dataset
import json

with open('../data/red_dwarf_episode_1.json', 'r') as f:
    script_data = json.load(f)
    
dialogues = [dialogue['text'] for dialogue in script_data['dialogues']]

tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')

tokenizer.pad_token = tokenizer.eos_token

tokenized_dialogues = tokenizer(
    dialogues, 
    padding=True, 
    truncation=True, 
    return_tensors="pt", 
    max_length=512
)

dataset = Dataset.from_dict(tokenized_dialogues)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_dataset.save_to_disk('train_data')
test_dataset.save_to_disk('test_data')

tokenizer.save_pretrained('distilgpt2_tokenizer')