import pandas as pd
import ast
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)


# ------------ 1. Load cleaned dataset ------------

df = pd.read_csv("data/cleaned_dataset.csv")

# ingredients_tokens -> list, then to plain text
def tokens_to_text(value):
    if isinstance(value, list):
        tokens = value
    elif isinstance(value, str):
        try:
            tokens = ast.literal_eval(value)
            if not isinstance(tokens, list):
                tokens = [value]
        except Exception:
            tokens = value.split()
    else:
        tokens = []
    return " ".join(str(t) for t in tokens)

df["ingredients_tokens"] = df["ingredients_tokens"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
df["ingredients_clean"] = df["ingredients_tokens"].apply(tokens_to_text)


# ------------ 2. Allergen detection  ------------

ALLERGEN_DICT = {
    "dairy": [
        "milk", "cheese", "butter", "cream", "yogurt",
        "lactose", "whey", "skim milk", "whole milk",
        "buttermilk", "condensed milk", "milk powder"
    ],
    "egg": ["egg", "egg yolk", "egg white"],
    "peanut": ["peanut", "peanut butter", "groundnut"],
    "tree_nut": [
        "almond", "walnut", "cashew", "hazelnut", "pistachio",
        "pecan", "macadamia", "brazil nut"
    ],
    "fish": ["salmon", "tuna", "cod", "trout", "haddock", "anchovy"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "scallop", "clam", "oyster"],
    "soy": ["soy", "soybean", "tofu", "soy sauce", "edamame", "soy milk"],
    "gluten": [
        "wheat", "flour", "white flour", "bread", "pasta",
        "barley", "rye", "spelt", "semolina"
    ]
}

def detect_allergens(text):
    text = text.lower()
    found = set()
    for allergen, words in ALLERGEN_DICT.items():
        for w in words:
            if w in text:
                found.add(allergen)
    return found

df["allergens"] = df["ingredients_clean"].apply(detect_allergens)

# Label: 0 = unsafe (has allergens), 1 = safe (no allergens)
df["is_safe"] = df["allergens"].apply(lambda s: 1 if len(s) == 0 else 0)

print("Label distribution:")
print(df["is_safe"].value_counts())


# ------------ 3. Train–test split ------------

X_text = df["ingredients_clean"]
y = df["is_safe"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.25, random_state=42, stratify=y
)

# ------------ 4. TF-IDF vectorization ------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1-Score (macro):  {f1:.4f}")

    # Confusion matrix (0 = unsafe, 1 = safe)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["unsafe", "safe"])
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

    return acc, f1


# ------------ 5. Logistic Regression ------------

logreg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear"
)

logreg_acc, logreg_f1 = evaluate_model(
    logreg, X_train, y_train, X_test, y_test, "Logistic Regression (safe/unsafe)"
)


# ------------ 6. Linear SVM ------------

svm = LinearSVC(class_weight="balanced")

svm_acc, svm_f1 = evaluate_model(
    svm, X_train, y_train, X_test, y_test, "Linear SVM (safe/unsafe)"
)