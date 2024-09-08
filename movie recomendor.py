import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def get_movie_recommendations():
    # Import data
    dataset = "https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv"
    df = pd.read_csv(dataset)

    # Preprocess data
    df_features = df[["Movie_Genre","Movie_Keywords","Movie_Tagline","Movie_Cast","Movie_Director"]].fillna("")
    x = df_features["Movie_Genre"] + " " + df_features["Movie_Keywords"] + " " + df_features["Movie_Tagline"] + " " + df_features["Movie_Cast"] + " " + df_features["Movie_Director"]

    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(x)

    # Calculate cosine similarity
    Similarity_score = cosine_similarity(x)

    # Get user input
    Favourite_Movie_Name = input("Enter your favourite movie name : ")
    All_Movies_Title_List = df["Movie_Title"].tolist()
    Movie_Recommendation = difflib.get_close_matches(Favourite_Movie_Name, All_Movies_Title_List)
    Close_Match = Movie_Recommendation[0]
    Index_of_Close_Matches_Movie = df[df.Movie_Title == Close_Match]["Movie_ID"].values[0]

    # Get recommendation score
    Recommendation_Score = list(enumerate(Similarity_score[Index_of_Close_Matches_Movie]))
    Sorted_Similar_Movies = sorted(Recommendation_Score, key = lambda x:x[1], reverse = True)

    # Print recommendations
    print("Top 30 Movies Suggested for You : \n")
    i = 1
    for movie in Sorted_Similar_Movies:
        index = movie[0]
        if i <= 30 and index != Index_of_Close_Matches_Movie:
            print(f"{i}. {df['Movie_Title'][index]}")
            i += 1

get_movie_recommendations()