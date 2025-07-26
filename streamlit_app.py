import streamlit as st
from app.recommender import load_data, build_similarity_matrix, recommend, fetch_poster
import pandas as pd
import os
from datetime import datetime

# --------------------- Initialization ---------------------

def initialize_logs():
    """Create log files if they don't exist."""
    os.makedirs("data", exist_ok=True)

    user_logs_path = "data/user_logs.csv"
    user_history_path = "data/user_history.csv"

    if not os.path.exists(user_logs_path):
        pd.DataFrame(columns=["timestamp", "movie_selected", "recommended_movies"]).to_csv(user_logs_path, index=False)

    if not os.path.exists(user_history_path):
        pd.DataFrame(columns=["user_id", "timestamp", "movie_selected"]).to_csv(user_history_path, index=False)

def log_recommendation(selected, recommendations):
    """Append a new recommendation log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "movie_selected": selected,
        "recommended_movies": ", ".join(recommendations)
    }
    pd.DataFrame([record]).to_csv("data/user_logs.csv", mode='a', header=False, index=False)

def log_user_history(user_id, selected_movie):
    """Append a new user activity log."""
    entry = {
        "user_id": user_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "movie_selected": selected_movie
    }
    pd.DataFrame([entry]).to_csv("data/user_history.csv", mode='a', header=False, index=False)

# --------------------- Load Data & Build Model ---------------------

df = load_data()
similarity_matrix = build_similarity_matrix(df)
initialize_logs()

# --------------------- Streamlit UI Setup ---------------------

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Netflix-Style Movie Recommender")
st.markdown("Select a movie you like and get 5 similar recommendations with posters!")

# Sidebar: User settings
st.sidebar.header("🧑 User Settings")
user_id = st.sidebar.text_input("Enter your name or ID:", value="guest")

# Sidebar: Watch history
st.sidebar.markdown("## 📽️ Your Watch History")
history_path = "data/user_history.csv"
if os.path.exists(history_path):
    hist_df = pd.read_csv(history_path)
    user_hist = hist_df[hist_df["user_id"] == user_id]
    if not user_hist.empty:
        for movie in user_hist.tail(5)["movie_selected"]:
            st.sidebar.write(f"👉 {movie}")
    else:
        st.sidebar.write("No history yet.")
else:
    st.sidebar.write("History file missing.")

# --------------------- Recommendation Engine ---------------------

movie_selected = st.selectbox("Choose a movie:", sorted(df['title'].unique()))

if st.button("🎯 Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(movie_selected, df, similarity_matrix)
        st.success("Here are your recommendations:")

        log_recommendation(movie_selected, recommendations)
        log_user_history(user_id, movie_selected)

        for title in recommendations:
            poster_url = fetch_poster(title)
            cols = st.columns([1, 4])
            with cols[0]:
                if poster_url:
                    st.image(poster_url, width=100)
                else:
                    st.write("❌ No poster")
            with cols[1]:
                st.markdown(f"**{title}**")

# Sidebar: Most popular movies
if st.sidebar.checkbox("🔥 Most Popular Movies Selected"):
    if os.path.exists(history_path):
        all_hist = pd.read_csv(history_path)
        top_movies = all_hist["movie_selected"].value_counts().head(5)
        st.sidebar.markdown("### 🏆 Top Picked Movies")
        for title, count in top_movies.items():
            st.sidebar.write(f"{title} — {count} times")

# Footer
st.markdown("---")
st.caption("🎥 Posters powered by [TMDB](https://www.themoviedb.org/)")
