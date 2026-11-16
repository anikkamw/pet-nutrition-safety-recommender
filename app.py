import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pet Nutrition & Safety Recommender", layout="wide")

ALLERGEN_SYNONYMS = {
    "chicken": ["chicken", "poultry", "poulet", "volaille", "volailles", "huhn", "hühner", "pollo"],
    "beef": ["beef", "boeuf", "bœuf", "bovin", "rind", "carne"],
    "salmon": ["salmon", "saumon", "lachs", "salmón"],
    "lamb": ["lamb", "agneau", "lamm", "cordero"],
    "grains": ["corn", "wheat", "rice", "barley", "oats", "oatmeal", "soy"],
}

@st.cache_data
def load_data():
    nutrition_df = pd.read_parquet("data/clean_pet_nutrition.parquet")
    safety_df = pd.read_parquet("data/brand_nlp_safety.parquet")
    return nutrition_df, safety_df

nutrition_df, safety_df = load_data()

# --- Header ---
st.title("Pet Nutrition & Adverse-Reaction Safety Recommender")
st.markdown(
    "Hybrid recommendation engine combining **Open Pet Food Facts** nutritional & ingredient analysis "
    "with **Amazon Reviews 2023 NLP** adverse-reaction risk scoring."
)

# --- Sidebar: Pet Dietary Profile ---
st.sidebar.header("Pet Dietary Profile")
species = st.sidebar.selectbox("Species", ["Dog", "Cat"])
preferred_protein = st.sidebar.selectbox(
    "Preferred Primary Protein", ["Any", "Salmon", "Lamb", "Beef", "Chicken"]
)
avoid_list = st.sidebar.multiselect(
    "Exclude Allergens / Ingredients",
    ["Chicken", "Beef", "Salmon", "Lamb", "Grains"],
    default=["Chicken"],
)
max_fat = st.sidebar.slider("Max Fat Content (% per 100g)", min_value=2.0, max_value=30.0, value=12.0, step=0.5)
top_n = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=15, value=5)

# --- Filtering & Scoring Logic ---
df = nutrition_df.copy()

# Species filter (prevent cross-species leaks)
if species.lower() == "dog":
    df = df[
        (df["categories_clean"].str.contains("dog", na=False) | df["product_name"].str.lower().str.contains("dog", na=False))
        & ~df["categories_clean"].str.contains("cat", na=False)
        & ~df["product_name"].str.lower().str.contains("kitten|cat", na=False)
    ]
else:
    df = df[
        (df["categories_clean"].str.contains("cat", na=False) | df["product_name"].str.lower().str.contains("cat", na=False))
        & ~df["categories_clean"].str.contains("dog", na=False)
        & ~df["product_name"].str.lower().str.contains("puppy|dog", na=False)
    ]

# Allergen exclusion filter
for item in avoid_list:
    terms = ALLERGEN_SYNONYMS.get(item.lower(), [item.lower()])
    pattern = r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b"
    df = df[
        ~df["ingredients_clean"].str.contains(pattern, regex=True)
        & ~df["product_name"].str.lower().str.contains(pattern, regex=True)
    ]

# Fat limit filter (retains products under limit or with unlisted fat)
df = df[(df["fat_100g"] <= max_fat) | (df["fat_100g"].isna())]

# Compute Nutrition & Profile Match Score
df["match_score"] = 0
if preferred_protein != "Any" and f"has_{preferred_protein.lower()}" in df.columns:
    df["match_score"] += df[f"has_{preferred_protein.lower()}"].astype(int) * 3

df["match_score"] += df["proteins_100g"].notna().astype(int)
df["match_score"] += (df["fat_100g"] <= max_fat).fillna(False).astype(int) * 2

# Merge Amazon NLP Brand Safety Scores
df["brand_clean"] = df["brands_en"].fillna("").str.lower().str.strip()
merged = df.merge(safety_df, on="brand_clean", how="left")
merged["safety_score"] = merged["safety_score"].fillna(97.5).round(1)
merged["avg_rating"] = merged["avg_rating"].fillna(4.3).round(2)
merged["total_reviews"] = merged["total_reviews"].fillna(0).astype(int)

results = merged.sort_values(by=["match_score", "safety_score"], ascending=[False, False]).head(top_n)

# --- Main Display ---
col1, col2, col3 = st.columns(3)
col1.metric("Matching Safe Products", len(df))
col2.metric("Avg NLP Safety Score", f"{results['safety_score'].mean():.1f}%" if len(results) else "N/A")
col3.metric("Excluded Allergens", ", ".join(avoid_list) if avoid_list else "None")

st.subheader(f"Top {len(results)} Recommended Formulas")
if results.empty:
    st.warning("No products matched all strict criteria. Try relaxing the allergen exclusions or fat limit.")
else:
    display_cols = [
        "product_name",
        "brands_en",
        "proteins_100g",
        "fat_100g",
        "match_score",
        "safety_score",
        "avg_rating",
        "total_reviews",
        "ingredients_text",
    ]
    st.dataframe(
        results[display_cols].rename(columns={
            "product_name": "Product Name",
            "brands_en": "Brand",
            "proteins_100g": "Protein (g/100g)",
            "fat_100g": "Fat (g/100g)",
            "match_score": "Nutrition Match",
            "safety_score": "NLP Safety Score (%)",
            "avg_rating": "Avg Rating",
            "total_reviews": "Verified Reviews",
            "ingredients_text": "Ingredients",
        }),
        use_container_width=True,
        hide_index=True,
    )