import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv('Data/movies.csv')
ratings = pd.read_csv('Data/ratings.csv') 

def viz_1():
    fig, ax = plt.subplots(figsize=(20,10))
    x = ratings.groupby('movieId').agg(lambda x: x.sum()/len(x))['rating']
    y = ratings.groupby('movieId').agg(sum)['rating']
    sns.scatterplot(x=x, y=y, alpha=.5)
    ax.set_xlabel('Avg Rating')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Average Movie Ratings by Number of Reviewers');
    
def viz_2():
    movie_df2 = movies.copy()
    movie_df2['genres'] = movie_df2['genres'].str.strip().str.split('|')
    genres_df = movie_df2.explode('genres')

    #initiate graph 
    fig, ax = plt.subplots(figsize = (30,10))


    #set the x and y parameters
    x = genres_df.groupby('genres')['movieId'].count().sort_values(ascending = False).index
    height = genres_df.groupby('genres')['movieId'].count().sort_values(ascending = False)

    #set axes labels
    ax.set_xlabel('Movie Categories',size = 20)
    ax.set_ylabel('Number of Movies',size = 20)
    ax.set_title('Categories by Popularity',size = 25)

    #display y-yabel and ticks on both left and right side
    plt.tick_params(labelright = True)
    ax.yaxis.set_ticks_position('both')

    #set the tick sizes for graph
    plt.yticks(size = 11)
    plt.xticks(rotation = 0, size = 11)

    #plit the bar graph of categories vs movies
    ax.bar(x,height);