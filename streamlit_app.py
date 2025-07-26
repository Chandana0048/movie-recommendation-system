import streamlit as st
from app.recommender import load_data, build_similarity_matrix, recommend, fetch_poster
import pandas as pd
import os
from datetime import datetime


def initialize_log():
    if not os.path.exists("data/user_logs.csv"):
        df = pd.DataFrame(columns=["timestamp", "movie_selected", "recommended_movies"])
        df.to_csv("data/user_logs.csv", index=False)

def log_recommendation(selected, recommendations):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "movie_selected": selected,
        "recommended_movies": ", ".join(recommendations)
    }
    df = pd.DataFrame([record])
    df.to_csv("data/user_logs.csv", mode='a', header=False, index=False)


# Load data and model
df = load_data()
similarity_matrix = build_similarity_matrix(df)

# ✅ Initialize log file
initialize_log()

# Streamlit UI setup
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Netflix-Style Movie Recommender")
st.markdown("Select a movie you like and get 5 similar recommendations with posters!")

# Dropdown for movie selection
movie_selected = st.selectbox("Choose a movie:", sorted(df['title'].unique()))

# On button click, recommend
if st.button("🎯 Recommend"):
    with st.spinner("Finding similar movies..."):
        results = recommend(movie_selected, df, similarity_matrix)
        st.success("Here are your recommendations:")

        for title in results:
            poster_url = fetch_poster(title)  # ✅ Corrected this line!
            cols = st.columns([1, 4])
            with cols[0]:
                if poster_url:
                    st.image(poster_url, width=100)
                else:
                    st.write("❌ No poster")
            with cols[1]:
                st.write(f"**{title}**")

        log_recommendation(movie_selected, results)


# Footer
st.markdown("---")
st.caption("🎥 Posters powered by [TMDB](https://www.themoviedb.org/)")
