from prepare_data import tokenizer, Dataset
import json

with open('../data/red_dwarf_episode_1.json', 'r') as f:
    script_data = json.load(f)
    
print(json.dumps(script_data, indent=4))

dialogues = [dialogue['text'] for dialogue in script_data['dialogues']]

tokenized_dialogues = tokenizer(
    dialogues, 
    padding=True, 
    truncation=True, 
    return_tensors="pt", 
    max_length=512
)

dataset = Dataset.from_dict(tokenized_dialogues)

print("Sample of tokenized data:")
for i in range(5):
    print(dataset[i])
    
train_test_split = dataset.train_test_split(test_size=0.1)
print("\nTrain dataset sample:")
print(train_test_split['train'][0])

print("\nTest dataset sample:")
print(train_test_split['test'][0])

print("\nTokenizer sample outputs:")
print(tokenizer.decode(tokenized_dialogues['input_ids'][0]))

