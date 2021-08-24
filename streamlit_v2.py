import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

st.title("Movie Recommendation")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ratings = pd.read_csv('Data/ratings.csv')
movies = pd.read_csv('Data/movies.csv')

st.write(movies)

# Define a TF-IDF Vectorizer Object.
tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(movies['genres'])
cosine_score = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)

#using TF-IDF Vectorizer Oject to calculate recommendations
def get_recommendations_based_on_genres(movie_title, cosine_score=cosine_score):
    """
    Calculates top 2 movies to recommend based on given movie titles genres. 
    :param movie_title: title of movie to be taken for base of recommendation
    :param cosine_sim_movies: cosine similarity between movies 
    :return: Titles of movies recommended to user
    """
    # Get the index of the movie that matches the title
    idx_movie = movies.loc[movies['title'].isin([movie_title])]
    idx_movie = idx_movie.index
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores_movies = list(enumerate(cosine_score[idx_movie][0]))
    
    # Sort the movies based on the similarity scores
    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores_movies = sim_scores_movies[0:4]
    
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores_movies]
    
    # Return the top 2 most similar movies
    return movies['title'].iloc[movie_indices]
    
#     # Return the top 2 most similar movies
#     print(movie_indices)

    
movie = st.text_input("Enter Movie Name",'Blue Sky (1994)')
if st.button('Enter'):
    result = movie.title()
    st.success(result)
    st.write(get_recommendations_based_on_genres(result))

#     movie_indices = [movies['title'].iloc[i[0]] for i in sim_scores_movies if movies['title'].iloc[i[0]]!=movie_title ]
    
#     # Return the top 2 most similar movies
#     print(movie_indices)