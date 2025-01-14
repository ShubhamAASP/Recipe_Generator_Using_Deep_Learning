import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load your custom tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/MyDrive/recipe_generator_project/tokenizer")

# Load the pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load your prepared dataset
dataset = load_dataset("text", data_files="/content/drive/MyDrive/recipe_generator_project/prepared_dataset/train.txt", split="train")

# Define training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/recipe_generator_project/results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/content/drive/MyDrive/recipe_generator_project/logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    no_cuda=not torch.cuda.is_available(),
    fp16=torch.cuda.is_available(),
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f) for f in data])}
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/content/drive/MyDrive/recipe_generator_project/recipe_generator")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")