import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
from rapidfuzz import fuzz, process

# =============== SPELL CHECK + FUZZY MATCH SETUP ===============

spell = SpellChecker()

def correct_spelling(text):
    words = text.lower().split()
    corrected = [spell.correction(w) for w in words]
    return " ".join(corrected)

def fuzzy_normalize(word, vocabulary, threshold=80):
    """
    Vraca najblizu rijec iz vokabulara ako fuzzy score >= threshold,
    u suprotnom vraca originalnu rijec.
    """
    match, score, _ = process.extractOne(word, vocabulary)
    return match if score >= threshold else word

def normalize_user_ingredients(text, vocabulary):
    """
    1) Spell-check nad unosom korisnika
    2) Fuzzy normalizacija prema TF-IDF vokabularu
    """
    words = text.lower().split()
    corrected = [spell.correction(w) for w in words]
    normalized = [fuzzy_normalize(w, vocabulary) for w in corrected]
    return " ".join(normalized)

# =============== NORMALIZACIJA TEKSTA / ENCODING FIX ===============

def normalize_text(text):
    """
    Popravlja:
    - mojibake tipa 'Â¼', 'Â½', 'â' itd. u prave razlomke
    - unicode escape tipa '\\u00bd' -> '½'
    """
    if not isinstance(text, str):
        return text

    t = text

    # 1) rucne zamjene najcescih sekvenci
    replacements = {
        "Â¼": "¼",
        "Â½": "½",
        "Â¾": "¾",
        "Â¹": "¹",
        "Â²": "²",
        "Â³": "³",
        "â“": "⅓",
        "â’": "⅔",
        "â•": "⅕",
        "â–": "⅖",
        "â—": "⅗",
        "â˜": "⅘",
        "â™": "⅙",
        "âš": "⅚",
    }
    for bad, good in replacements.items():
        t = t.replace(bad, good)

    # 2) probaj dekodirati unicode escape (\u00bd, \u00bc...)
    try:
        t = t.encode().decode("unicode_escape")
    except Exception:
        pass

    return t

# =============== 1. UČITAVANJE CLEANED DATASETA ===============

df = pd.read_csv("data/cleaned_dataset.csv")

# ===================== 2. TOKENI → TEKST =====================

df["ingredients_tokens"] = df["ingredients_tokens"].apply(ast.literal_eval)
df["ingredients_clean"] = df["ingredients_tokens"].apply(lambda tokens: " ".join(tokens))

# ===================== 3. ALERGENI =====================

ALLERGEN_DICT = {
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt"],
    "egg": ["egg"],
    "peanut": ["peanut", "peanut butter"],
    "tree_nut": ["almond", "walnut", "cashew", "hazelnut", "pistachio"],
    "fish": ["salmon", "tuna", "cod"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster"],
    "soy": ["soy", "soybean", "tofu", "soy sauce"],
    "gluten": ["wheat", "flour", "bread", "pasta", "barley", "rye"]
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

# ===================== 4. TF-IDF =====================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(df["ingredients_clean"])
VOCABULARY = vectorizer.get_feature_names_out()

# ===================== 5. FILTRIRANJE =====================

def filter_safe_recipes(user_allergens):
    """
    Filtrira recepte:
      - po kategorijama alergena (dairy, gluten, soy, ...)
      - po konkretnim sastojcima (tomato, onion, ...)
    """
    if not user_allergens:
        return df

    user_allergens = [a.strip().lower() for a in user_allergens if a.strip()]

    allergen_categories = set([a for a in user_allergens if a in ALLERGEN_DICT.keys()])
    allergen_terms = [a for a in user_allergens if a not in ALLERGEN_DICT.keys()]

    def is_safe(row):
        # 1) provjera kategorija (dairy, gluten, ...)
        if allergen_categories and (row["allergens"] & allergen_categories):
            return False

        # 2) provjera konkretnih sastojaka (tomato, peanut, ...)
        text = row["ingredients_clean"]
        for term in allergen_terms:
            if term in text:
                return False

        return True

    mask = df.apply(is_safe, axis=1)
    return df[mask]

# ===================== 6. REKOMENDACIJE =====================

def recommend_recipes(user_ingredients: str, user_allergens, top_k: int):
    """
    Vraća do top_k recepata (ako ih ima manje, vrati sve dostupne),
    sortirano:
      1) po broju pogodjenih sastojaka (vise je bolje)
      2) po cosine similarity (TF-IDF) kao tie-breaker
    """

    safe_df = filter_safe_recipes(user_allergens)

    if safe_df.empty:
        print("Nema sigurnih recepata za zadate alergije.")
        return safe_df

    # Spell-check + fuzzy normalizacija unosa sastojaka:
    user_ingredients_fixed = normalize_user_ingredients(user_ingredients, VOCABULARY)
    user_words = set(user_ingredients_fixed.split())

    if not user_words:
        print("Nisi unio nijedan sastojak.")
        return pd.DataFrame()

    # 1) izracunaj broj zajednickih sastojaka (overlap) za svaki recept
    def count_overlap(row):
        recipe_words = set(row["ingredients_clean"].split())
        overlap = user_words & recipe_words
        return len(overlap)

    safe_df = safe_df.copy()
    safe_df["overlap_count"] = safe_df.apply(count_overlap, axis=1)

    # izbacujemo recepte koji nemaju nijedan od navedenih sastojaka
    safe_df = safe_df[safe_df["overlap_count"] > 0]

    if safe_df.empty:
        print("Nema recepata koji sadrže ijedan od navedenih sastojaka.")
        return safe_df

    # 2) TF-IDF similarity samo na ovim receptima
    user_vec = vectorizer.transform([user_ingredients_fixed])

    safe_indices = safe_df.index
    safe_tfidf = tfidf_matrix[safe_indices]

    sims = cosine_similarity(user_vec, safe_tfidf)[0]
    safe_df["similarity"] = sims

    # 3) sortiranje:
    #   - prvo po overlap_count (vise zajednickih sastojaka je bolje)
    #   - zatim po similarity (veca slicnost je bolja)
    safe_df = safe_df.sort_values(
        by=["overlap_count", "similarity"],
        ascending=[False, False]
    )

    # uklanjanje duplikata
    safe_df = safe_df.drop_duplicates(subset=["recipe_title", "ingredients", "directions"])

    # ako traži 0 ili manje, ili više nego što ima – vrati sve
    if top_k <= 0 or top_k >= len(safe_df):
        return safe_df

    return safe_df.head(top_k)

# ===================== 7. ISPIS =====================

def print_recommendations(recs):
    if recs.empty:
        print("Nema preporučenih recepata.")
        return

    for idx, row in recs.iterrows():
        print("\n" + "="*60)
        print("RECEPT:", normalize_text(row["recipe_title"]))
        print("-"*60)
        print("SASTOJCI:")

        raw_ing = normalize_text(row["ingredients"])
        try:
            ing_list = ast.literal_eval(raw_ing)
            if isinstance(ing_list, list):
                for ing in ing_list:
                    print(" -", normalize_text(str(ing)))
            else:
                print(raw_ing)
        except Exception:
            print(raw_ing)

        print("\nNAČIN PRIPREME:")
        raw_dir = normalize_text(row["directions"])
        try:
            steps = ast.literal_eval(raw_dir)
            if isinstance(steps, list):
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {normalize_text(str(step))}")
            else:
                print(raw_dir)
        except Exception:
            print(raw_dir)

        print("\nBROJ POGOĐENIH SASTOJAKA:", row.get("overlap_count", "N/A"))
        print("SIMILARITY SCORE:", round(row["similarity"], 3))
        print("="*60)

# ===================== 8. INTERAKTIVNI DIO =====================

if __name__ == "__main__":
    print("\n=== Allergy-Aware Recipe Recommender ===\n")

    user_ingredients = input(
        "Unesi sastojke koje imas ili zelis koristiti (npr. 'chicken onion cucumber bread'): "
    ).strip()

    print(
        "\nMožeš unijeti alergije kao kategorije (dairy, gluten, soy...) "
        "ili kao sastojke (tomato, milk, peanut)."
    )

    allergens_input = input(
        "Unesi na šta si alergičan (odvoji zarezom, npr. 'tomato, milk'): "
    ).strip()

    if allergens_input:
        user_allergens = [a.strip() for a in allergens_input.split(",") if a.strip()]
    else:
        user_allergens = []

    try:
        top_k = int(input("\nKoliko recepata zelis da prikazem? (npr. 5): ").strip())
    except ValueError:
        top_k = 5

    print("\nTražim recepte...\n")

    recs = recommend_recipes(
        user_ingredients=user_ingredients,
        user_allergens=user_allergens,
        top_k=top_k
    )

    print_recommendations(recs)
