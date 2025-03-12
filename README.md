
# Recipe Generator Using Deep Learning (GPT-2)

## Overview

This project presents a deep learning-based **Recipe Generator** that leverages the power of the **GPT-2 model** to generate creative and coherent recipes.
The system is built with the help of **Google Colab** and includes an **Exploratory Data Analysis (EDA)** process to prepare and analyze the dataset. The model can generate unique recipes based on user input, offering an innovative solution for culinary inspiration and creativity.

## Key Features

- **GPT-2 Model**: Utilizes the pre-trained GPT-2 model to generate recipe content based on provided prompts.
- **Recipe Generation**: The system generates recipes, including ingredients, preparation methods, and serving suggestions.
- **Data Preprocessing & EDA**: Includes data cleaning, exploration, and analysis of the recipe dataset to ensure high-quality inputs for the model.
- **Google Colab Integration**: Fully implemented in Google Colab for easy access and execution on the cloud.

## Technologies Used

- **Deep Learning**: GPT-2 (Generative Pre-trained Transformer 2)
- **Python**: Main programming language
- **Libraries**:
  - TensorFlow/PyTorch (for model training and inference)
  - Pandas (for data processing)
  - Matplotlib/Seaborn (for EDA and data visualization)
  - Transformers (Hugging Face) for GPT-2 integration
- **Google Colab**: Cloud-based platform for running the code and experimenting with the model.

## Dataset

The dataset used for training the GPT-2 model consists of recipes with details about ingredients, cooking instructions, cuisine types, and preparation times. The dataset has been carefully cleaned and preprocessed to ensure high-quality inputs for the model. The EDA process is used to analyze the dataset and uncover patterns or insights for model improvement.

## EDA Process

The Exploratory Data Analysis (EDA) process is crucial for understanding the dataset and preparing it for model training. It involves:

1. **Data Cleaning**: Removing any duplicates, irrelevant columns, or incorrect data entries.
2. **Statistical Summary**: Descriptive analysis of the dataset to identify key features such as ingredient frequency, recipe length, and more.
3. **Data Visualization**: Visualizing data distributions, ingredient popularity, and other metrics to gain insights into the dataset.
4. **Feature Engineering**: Extracting useful features to improve model performance.

## Installation

### Prerequisites

- **Python 3.x** or higher
- **Google Colab** account for running the code in the cloud.
- **Required Libraries**: All necessary libraries are listed in the `requirements.txt` file.

### Steps to Run

1. Clone the repository to your local machine or directly use Google Colab.
2. Install required libraries using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. Upload your dataset and run the notebook from Google Colab.
4. Follow the notebook instructions to train the GPT-2 model or use the pre-trained model to generate recipes.

### Usage

1. **Dataset Preprocessing**: Start by cleaning and preparing your recipe dataset.
2. **Model Training**: Train the GPT-2 model on the recipe dataset using the provided notebook.
3. **Recipe Generation**: After training, input a prompt (e.g., a cuisine or dish type), and the model will generate a recipe complete with ingredients and instructions.

### Example

```python
prompt = "Generate a vegan recipe for dinner"
recipe = generate_recipe(prompt)
print(recipe)
```


## Results

- The model successfully generates realistic and coherent recipes based on given prompts.
- Recipes include ingredients, quantities, and step-by-step instructions.
- The model can be further fine-tuned to improve the specificity of recipes or add more detailed cuisine types.

## Future Work

- **Fine-Tuning for Specific Cuisines**: Train the model specifically for various cuisines such as Italian, Mexican, etc.
- **Advanced Filtering**: Implement functionality for filtering recipes based on dietary requirements (e.g., vegan, gluten-free).
- **User Interface**: Develop a user-friendly interface for inputting prompts and displaying generated recipes.

