import re
from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

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

def generate_recipe(ingredients, max_length=400, temperature=0.7):
    prompt = f"Create a detailed recipe using these ingredients: {ingredients}\n\nRecipe:"
    
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
    return format_recipe(generated_text, ingredients)

def format_recipe(text, original_ingredients):
    # Remove the initial prompt from the generated text
    recipe_text = text.split("Recipe:", 1)[-1].strip()
    
    # Split the recipe into sections
    sections = re.split(r'\n(?=\w+:)', recipe_text)
    
    formatted_recipe = ""
    for section in sections:
        if ':' in section:
            title, content = section.split(':', 1)
            formatted_recipe += f"{title.strip()}:\n{content.strip()}\n\n"
        else:
            formatted_recipe += f"{section.strip()}\n\n"
    
    return formatted_recipe.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        recipe = generate_recipe(ingredients)
        return render_template('index.html', recipe=recipe)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)