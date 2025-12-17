import random
import numpy as np

import modelv2 


# ---------- 1. Helper: make a query from a recipe ----------

def create_query_from_recipe(row, max_ingredients=5):
    """
    Create a synthetic user query by sampling up to max_ingredients
    tokens from ingredients_clean (already preprocessed in model.py).
    """
    text = row["ingredients_clean"]
    if not isinstance(text, str):
        return None

    tokens = list(set(text.split()))
    if not tokens:
        return None

    k = min(max_ingredients, len(tokens))
    chosen = random.sample(tokens, k)
    return " ".join(chosen)


# ---------- 2. Retrieval metrics: Top-1, Hit@k, Precision@k, Recall@k ----------

def evaluate_retrieval(top_k=5, num_queries=200, random_state=42):
    """
    Evaluate the recommendation quality of model.recommend_recipes
    using synthetic queries built from existing recipes.

    Metrics:
      - Top-1 Accuracy
      - Hit@k (Success@k)
      - Precision@k
      - Recall@k (with 1 relevant recipe per query)
    """
    random.seed(random_state)

    df = modelv2.df  # from modelv2.py
    indices = df.sample(n=min(num_queries, len(df)), random_state=random_state).index

    top1_hits = 0
    hit_at_k = 0
    precision_at_k_list = []
    recall_at_k_list = []

    evaluated_queries = 0

    for idx in indices:
        row = df.loc[idx]
        query = create_query_from_recipe(row)
        if not query:
            continue

        # no allergens specified for retrieval quality eval
        recs = modelv2.recommend_recipes(
            user_ingredients=query,
            user_allergens=[],
            top_k=top_k
        )

        # if no recommendations, skip this query
        if recs is None or recs.empty:
            continue

        evaluated_queries += 1

        # rank of the original recipe in the results
        if idx in recs.index:
            rank = list(recs.index).index(idx) + 1  # ranks are 1-based
        else:
            rank = None

        # Top-1 Accuracy
        if rank == 1:
            top1_hits += 1

        # Hit@k (Success@k)
        if rank is not None and rank <= top_k:
            hit_at_k += 1
            # only one relevant recipe -> precision@k = 1/k if hit, else 0
            precision_at_k_list.append(1.0 / top_k)
            recall_at_k_list.append(1.0)  # recall = 1 if we found the only relevant recipe
        else:
            precision_at_k_list.append(0.0)
            recall_at_k_list.append(0.0)

    if evaluated_queries == 0:
        print("No queries could be evaluated (no recommendations returned).")
        return None

    top1_acc = top1_hits / evaluated_queries
    hitk = hit_at_k / evaluated_queries
    precision_k = float(np.mean(precision_at_k_list))
    recall_k = float(np.mean(recall_at_k_list))

    print("\n=== RETRIEVAL EVALUATION (k = {}) ===".format(top_k))
    print(f"Evaluated queries:   {evaluated_queries}")
    print(f"Top-1 Accuracy:      {top1_acc:.4f}")
    print(f"Hit@{top_k}:          {hitk:.4f}")
    print(f"Precision@{top_k}:    {precision_k:.4f}")
    print(f"Recall@{top_k}:       {recall_k:.4f}")

    return {
        "evaluated_queries": evaluated_queries,
        "top1_accuracy": top1_acc,
        "hit_at_k": hitk,
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
    }


# ---------- 3. Allergy violation rate ----------

def evaluate_allergy_violation(top_k=5, num_tests=200, random_state=123):
    """
    Evaluate how often the recommender still returns recipes that
    contain an allergen the user says they are allergic to.

    Steps:
      1. Sample recipes that contain at least one allergen.
      2. Choose one allergen category from that recipe as the "user allergy".
      3. Build a query from that recipe's ingredients.
      4. Call recommend_recipes with that allergy.
      5. Check how many returned recipes still contain that allergen.
    """
    random.seed(random_state)

    df = modelv2.df

    # choose recipes that actually have allergens
    allergen_recipes = df[df["allergens"].apply(lambda s: len(s) > 0)]
    if allergen_recipes.empty:
        print("Dataset has no detected allergens, cannot evaluate allergy violations.")
        return None

    indices = allergen_recipes.sample(
        n=min(num_tests, len(allergen_recipes)),
        random_state=random_state
    ).index

    total_recs = 0
    violations = 0
    tested_queries = 0

    for idx in indices:
        row = df.loc[idx]
        if not row["allergens"]:
            continue

        # pick one allergen category from this recipe
        chosen_allergen = list(row["allergens"])[0]

        query = create_query_from_recipe(row)
        if not query:
            continue

        recs = modelv2.recommend_recipes(
            user_ingredients=query,
            user_allergens=[chosen_allergen],
            top_k=top_k
        )

        if recs is None or recs.empty:
            continue

        tested_queries += 1

        for _, r in recs.iterrows():
            total_recs += 1
            # violation if recipe still has that allergen category
            if chosen_allergen in r["allergens"]:
                violations += 1

    if total_recs == 0:
        print("No recommendations generated during allergy evaluation.")
        return None

    violation_rate = violations / total_recs

    print("\n=== ALLERGY FILTER EVALUATION (k = {}) ===".format(top_k))
    print(f"Tested queries:          {tested_queries}")
    print(f"Total recommended items: {total_recs}")
    print(f"Violating recommendations: {violations}")
    print(f"Allergy-violation rate:  {violation_rate:.4f}")

    return {
        "tested_queries": tested_queries,
        "total_recs": total_recs,
        "violations": violations,
        "violation_rate": violation_rate,
    }


# ---------- 4. Run when script is executed directly ----------

if __name__ == "__main__":
    # Evaluate retrieval quality for k = 3 and k = 5, for example
    evaluate_retrieval(top_k=3, num_queries=200)
    evaluate_retrieval(top_k=5, num_queries=200)

    # Evaluate allergy violation rate
    evaluate_allergy_violation(top_k=5, num_tests=200)
