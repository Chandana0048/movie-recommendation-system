from recommender import load_data, build_similarity_matrix, recommend

df = load_data()
sim = build_similarity_matrix(df)

movie = input("Enter a movie title: ")
results = recommend(movie, df, sim)

print("\nRecommendations:")
for r in results:
    print("-", r)
