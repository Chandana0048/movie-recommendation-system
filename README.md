# 🎬 Netflix-Style Movie Recommender




A content-based movie recommendation system built using Python and scikit-learn.

## 🚀 Features
- Input a movie title and get 5 similar recommendations
- Built using genres and cosine similarity
- CLI-based MVP (streamlit UI coming next)

## 🛠 Tech Stack
- Python
- pandas
- scikit-learn
- Streamlit (soon)







🧠 PHASE 1: Project Architecture & Planning
🗂 Project Folder Structure:
cpp
Copy
Edit
movie-recommender/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── links.csv
│
├── notebooks/
│   └── recommendation_engine.ipynb
│
├── app/
│   ├── main.py
│   ├── recommender.py
│   └── utils.py
│
├── static/
│   └── posters/
│
├── templates/ (if using Flask)
│   └── index.html
│
├── README.md
├── requirements.txt
└── streamlit_app.py (optional)

📦 PHASE 2: Dataset Setup
🔗 Download MovieLens 100k:
https://grouplens.org/datasets/movielens/100k/

Place movies.csv and ratings.csv into the data/ folder.

🔧 PHASE 3: Build Content-Based Recommendation Engine
✅ What is it?
Recommends movies based on similar features — genre, plot keywords, etc.

💥 Tools:
pandas

scikit-learn → CountVectorizer, cosine_similarity

👨‍💻 Step-by-Step:
python
Copy
Edit
# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("data/movies.csv")
    # If using genres as a feature
    df["features"] = df["genres"].fillna('')
    return df

def build_similarity_matrix(df):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df["features"])
    similarity = cosine_similarity(count_matrix)
    return similarity

def recommend(title, df, similarity):
    if title not in df['title'].values:
        return ["Movie not found"]
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sorted_scores]
    return df['title'].iloc[movie_indices].tolist()

🔁 PHASE 4: Collaborative Filtering with Surprise Library
✅ What is it?
Recommends movies based on similar user ratings — very real-world!

💥 Tools:
surprise library (install via pip install scikit-surprise)

👨‍💻 Code Example:
python
Copy
Edit
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def build_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_builtin('ml-100k')  # or load from your CSV
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    print("RMSE:", accuracy.rmse(predictions))
    return model

def predict_rating(model, user_id, movie_id):
    return model.predict(user_id, movie_id).est
🖼️ PHASE 5: Frontend using Streamlit (Optional but 🔥🔥)
🚀 Install:
bash
Copy
Edit
pip install streamlit
✨ Basic UI:
python
Copy
Edit
# streamlit_app.py
import streamlit as st
from app.recommender import load_data, build_similarity_matrix, recommend

df = load_data()
similarity = build_similarity_matrix(df)

st.title("🎬 Movie Recommender")
movie = st.selectbox("Pick a movie", df["title"].values)

if st.button("Recommend"):
    recs = recommend(movie, df, similarity)
    st.write("Recommended:")
    for r in recs:
        st.write(r)
📊 PHASE 6: Visualization Add-ons
Optional but makes your resume 💎:

Use matplotlib or seaborn to show:

Genre distribution

User rating behavior

Heatmap of movie correlations

📖 PHASE 7: Documentation
README.md must include:
markdown
Copy
Edit
# Netflix-style Movie Recommendation System 🎥🍿

## What it does
This project recommends similar movies based on genres and user ratings using content-based and collaborative filtering.

## Tech Stack
- Python, pandas, scikit-learn, surprise, Streamlit

## How to Run
1. Clone this repo
2. Install requirements
3. Run with: `streamlit run streamlit_app.py`

## Sample Screenshot
(insert GIF or PNG)

## Author
Chandana KP - [LinkedIn](https://www.linkedin.com/in/chandana-puttanagappa)
✍️ PHASE 8: Blog It (Optional But Powerful)
Write on Medium:

Title: “I Built a Netflix-Style Recommender System: Here's How You Can Too”

Break down:

Intro + problem

Two types of recommendation logic

Code + visuals

What you learned

Why it's useful for real products


