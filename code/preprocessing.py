import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# POS tags mappings
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return NOUN  # default



# dataset read
df = pd.read_csv("data/dataset.csv")


df = df[['recipe_title', 'ingredients', 'directions']]

# lowercasing
df['recipe_title'] = df['recipe_title'].str.lower()
df['ingredients'] = df['ingredients'].str.lower()
df['directions'] = df['directions'].str.lower()

# tokenization
df['recipe_name_tokens'] = df['recipe_title'].apply(word_tokenize)
df['ingredients_tokens'] = df['ingredients'].apply(word_tokenize)
df['directions_tokens'] = df['directions'].apply(word_tokenize)

# punctu. removal 
def remove_punctuation(tokens):
    return [t for t in tokens if t.isalnum()]

df['recipe_name_tokens'] = df['recipe_name_tokens'].apply(remove_punctuation)
df['ingredients_tokens'] = df['ingredients_tokens'].apply(remove_punctuation)
df['directions_tokens'] = df['directions_tokens'].apply(remove_punctuation)


# stopword removal
stop_words = set(word.lower() for word in stopwords.words('english'))



def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

df['recipe_name_tokens'] = df['recipe_name_tokens'].apply(remove_stopwords)
df['ingredients_tokens'] = df['ingredients_tokens'].apply(remove_stopwords)
df['directions_tokens'] = df['directions_tokens'].apply(remove_stopwords)



# lemmatization
lemmatizer = WordNetLemmatizer()

# lemmatizaon

def lemmatize_tokens(tokens):
    pos_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(t, get_wordnet_pos(pos)) for t, pos in pos_tokens]


df['recipe_name_tokens'] = df['recipe_name_tokens'].apply(lemmatize_tokens)
df['ingredients_tokens'] = df['ingredients_tokens'].apply(lemmatize_tokens)
df['directions_tokens'] = df['directions_tokens'].apply(lemmatize_tokens)



# save to new file 
df.to_csv("data/cleaned_dataset.csv", index=False)

