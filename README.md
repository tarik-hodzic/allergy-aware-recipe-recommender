<h1>Allergy-Aware Recipe Recommender</h1>

Allergy-Aware Recipe Recommender is an intelligent system designed to help users find suitable recipes
based on food allergies, dietary restrictions, and available ingredients.  
The project applies Natural Language Processing (NLP) techniques to analyze recipe data and generate
safe, personalized recommendations.

<h2>Technologies Used</h2>

- Python  
- pandas  
- NLTK (tokenization, stop-word removal, POS tagging, lemmatization)  
- scikit-learn (TF-IDF, cosine similarity, Logistic Regression, Linear SVM, train/test split)  
- pyspellchecker (spell correction)  
- RapidFuzz (fuzzy matching)  
- matplotlib (evaluation and confusion matrix visualization)    

<h2>Project Purpose</h2>

This project was built to:

- Design an allergy-aware recipe recommendation system  
- Apply Natural Language Processing techniques to real-world text data  
- Analyze and preprocess large recipe datasets  
- Detect and filter allergens from ingredient lists  
- Generate personalized recipe recommendations based on user constraints  

<h2>Key Features</h2>

- Allergy-aware recipe filtering  
- Ingredient-based recommendation logic  
- Text preprocessing (tokenization, lemmatization, stop-word removal)  
- Ingredient and allergen detection based on NLP preprocessing and text matching 
- Similarity-based recipe recommendations using TF-IDF  
- Support for user-defined allergies and dietary preferences  

<h2>Dataset</h2>

The project uses a public recipe dataset containing ingredient lists and cooking instructions.

Dataset link:  
https://www.kaggle.com/datasets/prashantsingh001/recipes-dataset-64k-dishes

<h2>How It Works</h2>

- Recipe data is loaded from the dataset  
- Text data is cleaned and preprocessed using NLP techniques  
- Allergens are detected based on user input  
- Recipes containing unsafe ingredients are filtered out  
- TF-IDF vectorization is applied to represent recipes numerically  
- Similarity metrics are used to recommend the most relevant safe recipes  

<h2>Example Use Case</h2>

- A user specifies allergies (e.g. peanuts, milk, gluten)  
- Recipes containing these allergens are excluded  
- Remaining recipes are ranked by similarity and relevance  
- The user receives safe and personalized recipe recommendations  

<h2>Project Scope</h2>

This project focuses on demonstrating the application of NLP and recommendation techniques
in health-aware systems rather than production-level deployment.

<h2>Authors</h2>

Created by Tarik Hodžić, Naim Pjanić, Ammarudin Kovačević
