import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)

def load_and_prepare_data(file_path, max_length=512):
    # Load the CSV file
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} rows from CSV")

    # Combine input columns, ensuring all values are strings
    df['input_text'] = (df['RecipeName'].astype(str) + ' | ' + 
                        df['Ingredients'].astype(str) + ' | ' + 
                        df['Cuisine'].astype(str) + ' | ' + 
                        df['Course'].astype(str) + ' | ' + 
                        df['Diet'].astype(str))
    
    # Ensure Instructions are strings
    df['Instructions'] = df['Instructions'].astype(str)

    # Remove any rows with empty input_text or Instructions
    df = df.dropna(subset=['input_text', 'Instructions'])
    df = df[df['input_text'].str.strip() != '']
    df = df[df['Instructions'].str.strip() != '']
    logging.info(f"After cleaning, {len(df)} rows remain")

    # Prepare the dataset
    dataset = Dataset.from_pandas(df[['input_text', 'Instructions']])
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples['input_text'], padding='max_length', truncation=True, max_length=max_length)
        outputs = tokenizer(examples['Instructions'], padding='max_length', truncation=True, max_length=max_length)
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': outputs.input_ids
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    logging.info("Dataset tokenized")

    # Split the dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    logging.info(f"Train set size: {len(split_dataset['train'])}")
    logging.info(f"Test set size: {len(split_dataset['test'])}")

    return split_dataset, tokenizer

if __name__ == "__main__":
    file_path = 'IndianFoodDatasetCSV.csv'  # Update this to your CSV file path
    dataset, tokenizer = load_and_prepare_data(file_path)
    tokenizer.save_pretrained('./tokenizer')
    dataset.save_to_disk('./prepared_dataset')
    logging.info("Data preparation completed")