import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model from the checkpoint
model_path = "ingredient_based_model/checkpoint-10000"
model = GPT2LMHeadModel.from_pretrained(model_path)

# Load the original GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set pad_token_id to eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# Check if CUDA is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded from checkpoint and moved to {device}")

def generate_recipe(ingredients, max_length=450, temperature=0.7):
    prompt = f"Ingredients: {ingredients}\n\nRecipe:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()

# Interactive loop for generating recipes
while True:
    ingredients = input("\nEnter ingredients (or 'quit' to exit): ")
    if ingredients.lower() == 'quit':
        break
    generated_recipe = generate_recipe(ingredients)
    print("\nGenerated Recipe:")
    print(generated_recipe)