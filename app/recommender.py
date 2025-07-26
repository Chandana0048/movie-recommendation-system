import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re

# --- TMDB API KEY ---
TMDB_API_KEY = "your key here"

# --- Step 1: Load and Prepare Dataset ---
def load_data():
    df = pd.read_csv("data/movies.csv")

    # Remove year from title like "Titanic (1997)" → "Titanic"
    df['title'] = df['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    # Reformat titles like "Postman, The" → "The Postman"
    def normalize_title(title):
        title = title.strip()
        if "," in title:
            parts = title.split(",")
            if len(parts) == 2 and parts[1].strip().lower() in ["the", "a", "an"]:
                return f"{parts[1].strip()} {parts[0].strip()}"
        return title

    df['title'] = df['title'].apply(normalize_title)

    # Create feature string from genres for content-based filtering
    df["features"] = df["genres"].str.replace("|", " ", regex=False)

    return df

# --- Step 2: Build Similarity Matrix ---
def build_similarity_matrix(df):
    vectorizer = CountVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["features"].fillna(''))
    similarity = cosine_similarity(vectors)
    return similarity

# --- Step 3: Recommend Movies ---
def recommend(title, df, similarity, top_n=5):
    title = title.strip().lower()
    matched_titles = df[df['title'].str.lower().str.contains(title)]

    if matched_titles.empty:
        return ["Movie not found"]

    idx = matched_titles.index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sorted_scores]
    return df['title'].iloc[movie_indices].tolist()

# --- Step 4: Clean Movie Title for TMDB API ---
def clean_title(title):
    # Remove anything in parentheses
    title = re.sub(r'\(.*?\)', '', title).strip()

    # Convert "Movie, The" to "The Movie"
    if "," in title:
        parts = title.split(",")
        if len(parts) == 2 and parts[1].strip().lower() in ["the", "a", "an"]:
            title = f"{parts[1].strip()} {parts[0].strip()}"

    return title

# --- Step 5: Fetch Poster from TMDB ---
def fetch_poster(movie_title):
    
    cleaned_title = clean_title(movie_title)
    print(f"🔍 Searching poster for cleaned title: '{cleaned_title}'")

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={cleaned_title}"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                # Try exact match first
                for result in data['results']:
                    if result['title'].lower().strip() == cleaned_title.lower().strip():
                        poster_path = result.get('poster_path')
                        if poster_path:
                            return f"https://image.tmdb.org/t/p/w500{poster_path}"

                # Fallback to first result
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
            break  # success

        except requests.exceptions.RequestException as e:
            print(f"🔁 Retry {attempt + 1} failed for '{movie_title}': {e}")
            continue

    print(f"❌ Failed to fetch poster for '{movie_title}' after retries")
    return "https://via.placeholder.com/100x150?text=No+Image"
