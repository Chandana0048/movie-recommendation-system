import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("data/movies.csv")
    df['title'] = df['title'].str.replace(r"\(\d+\)", "", regex=True)
    df["features"] = df["genres"].str.replace("|", " ")
    return df

def build_similarity_matrix(df):
    vectorizer = CountVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["features"].fillna(''))
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(title, df, similarity, top_n=5):
    # Normalize the title input
    title = title.strip().lower()
    
    # Try to match title case-insensitively
    matched_titles = df[df['title'].str.lower().str.contains(title)]

    if matched_titles.empty:
        return ["Movie not found"]

    # Take the first matched movie
    idx = matched_titles.index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sorted_scores]
    return df['title'].iloc[movie_indices].tolist()

