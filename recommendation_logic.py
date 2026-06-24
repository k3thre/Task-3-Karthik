import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# PHASE 1: INPUT (Load Dataset & Ingest User Profile)
# =====================================================================
print("--- PHASE 1: INGESTING DATA & USER INPUT ---")

# 1. Load the dataset
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'raw_skills.csv')
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. {len(df)} target job roles available.")
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found.")
    exit()

# 2. Onboarding Survey / User Input (Ingests raw descriptive profile tags)
print("\n--- Enter your details for the Onboarding Survey ---")
skill_1 = input("Enter Skill/Interest 1: ").strip()
skill_2 = input("Enter Skill/Interest 2: ").strip()
skill_3 = input("Enter Skill/Interest 3: ").strip()

# Combine user inputs into a single text profile string
user_profile = f"{skill_1} {skill_2} {skill_3}"
print(f"\nCreated User Profile String: '{user_profile}'\n")


# =====================================================================
# PHASE 2: PROCESS (TF-IDF Vectorization & Cosine Similarity)
# =====================================================================
print("--- PHASE 2: PROCESSING ALGORITHMIC SIMILARITY ---")

# 1. Combine the item skills with the user profile to build a shared vocabulary space
corpus = list(df['skills']) + [user_profile]

# 2. Apply TF-IDF Feature Extraction (Penalizes generic words, rewards descriptive terms)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# 3. Separate the item vectors from the user profile vector
item_vectors = tfidf_matrix[:-1]   # Rows representing the dataset roles
user_vector = tfidf_matrix[-1]     # The final row representing the user profile

# 4. Compute Cosine Similarity (Calculates the mathematical angle alignment)
similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()
print("Vector mapping and Cosine Similarity computations complete.\n")


# =====================================================================
# PHASE 3: OUTPUT (Sorting & Filtering Top 3 Recommendations)
# =====================================================================
print("--- PHASE 3: EVALUATING & SORTING OUTPUT ---")

# Append calculated alignment scores back to our dataframe
df['similarity_score'] = similarity_scores

# Sort available items in descending order based on similarity scores
ranked_df = df.sort_values(by='similarity_score', ascending=False)

# Truncate output to return a tailored Top-3 list
top_3 = ranked_df.head(3)

print("\n🎯 [Top-3 Recommendations for You] 🎯")
print("==================================================")
for index, row in top_3.iterrows():
    print(f"🏆 Role: {row['role']}")
    print(f"   Alignment Score: {row['similarity_score']:.4f}")
    print(f"   Required Profile: {row['skills']}")
    print("--------------------------------------------------")