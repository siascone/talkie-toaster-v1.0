from transformers import DistilGPT2Tokenizer, DistilGPT2ForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk

model = DistilGPT2ForCausalLM.from_pretrained('distilgpt2')
tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')

model.resize_token_embeddings(len(tokenizer))

training_dataset = load_from_disk("./train_data/data-00000-of-00001.arrow")
testing_dataset = load_from_disk("./test_data/data-00000-of-00001.arrow")
training_arguments = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=training_dataset,
    eval_dataset=testing_dataset,
    tokenizer=tokenizer
)

trainer.train()

