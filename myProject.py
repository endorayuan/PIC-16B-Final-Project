#Importing all necessary libraries
import pandas as pd
import numpy as np
from numpy import random
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import sys
import warnings
import os

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture

import streamlit as st
from difflib import get_close_matches

#PREPROCESSING________________________________________
def standardization(input_data):
  '''
  This function standardizes input data by removing punctuation and lowercasing letters.
  Args:
    input_data (panda series): series containing text
  Return:
    standardized_text (panda series): series containing standardized text
  '''
  #removes punctuation from the text and replaces it with ' '
  no_punctuation = input_data.str.replace(r'[^\w\s]', '', regex=True)
  #lowercases all uppercase letters
  standardized_text = no_punctuation.str.lower()
  return standardized_text

def list_to_string(x):
  '''
  This function converts a list into a long string
  Args:
    x (list): list of strings
  Return:
    x (string): long string containing each string in the list
  '''
  #Turns list into string
  if isinstance(x, list):
    return " ".join(x)
  return str(x)

def preprocessing(df):
  '''
  This function preprocesses the TMDB_movie pandas data frame
  Args:
    df (pandas data frame): data frame containing movies
  Return:
    df (pandas data frame): preprocessed data frame
  '''
  df = df.copy()

  #Removing explicit content
  df = df[df["adult"] == False]

  #Dropping irrelevant columns 
  df = df.drop(columns = ['vote_count', 'runtime', 'backdrop_path', 'budget', 'homepage', 'original_language', 'original_title',
                                  'production_companies', 'production_countries', 'spoken_languages'])

  #Creating a new column called release year
  df["release_year"]= pd.to_datetime(df["release_date"]).dt.year
  df["display_title"] = (df["title"]+" ("+df["release_year"].astype(str)+")")

  #Selecting movies that have been released on or after 1980
  df = df[df["status"]=="Released"]
  df = df[df["release_year"]>=1980]

  #Removing all movies with no values for keywords, genre, and overview
  df = df.dropna(subset = ["genres", "keywords", "overview"])

  df["genres"] = df["genres"].apply(list_to_string)
  df["keywords"] = df["keywords"].apply(list_to_string)

  #Combines columns for Text Vectorization
  df["columns"] = df["keywords"] + " " + df["genres"] + " " + df["overview"] # combines columns

  #Standardizes Columns
  df["columns"] = standardization(df["columns"])
  df = df.reset_index()
  return df

# LOADING THE DATASET ________________________________________
# use this cache function so streamlit doesn't run the entire script for every interaction
@st.cache_data
def load_data():
    df = pd.read_csv("TMDB_movie_dataset_v11.csv")
    return preprocessing(df)

df = load_data()

#USER INTERFACE + user input function _______________________________

# displaying arguments
st.title("Recommendation Box")
st.write("Made by Vy, Ashley, and Endora")
st.write("Hello and welcome to the recommendation box!")
st.write("This is a movie recommendation system that utilizes machine learning to help you decide which movie to watch next.")
st.write("In order to use this program, type in one movie you've watched recently that you loved and want to see more of!")

user_input = st.text_input("Your Movie Here")
st.write("Make sure it's the correct spelling & capitalization!")

if user_input:
    user_movies = [movie.strip() for movie in user_input.split(",")]
    correct_movies= []

    # Iterates through the users movie
    for movie in user_movies:
        #gets the closest title in the dataframe
        matches = get_close_matches(movie, df["display_title"],cutoff=0.4)
        #If there is a match
        if matches:
            #Appends the first matched movie title into correct_movies
            correct_movies.append(matches[0])
        else:
            st.warning(f"No movie found for: {movie}")

    if correct_movies:
      st.write("We think you'd enjoy the following movie(s)...")
      st.success(f"Matched movies: {correct_movies}")
      recommendations = recommend(correct_movies[0], word2vec_vector)
      st.dataframe(recommendations)

#DATA VISUALIZATION___________________________________

#Creates a histogram of movies released each year
fig, ax=plt.subplots(figsize=(20,8))
fig= sns.histplot(df, x= "release_year")
plt.xticks(np.arange(df["release_year"].min(),df["release_year"].max()+1,10)) #this was to change the tiks bcuz it was horrific
plt.suptitle("Representation of the Growing Influx of Movies",fontsize=20, fontweight = "bold")
plt.ylabel("Number of Movies", fontweight = "bold")
plt.xlabel("Release Year", fontweight = "bold")
plt.show()

#Creates a line plot of the average movie rating per year
fig= plt.figure(figsize=(20,8))
average_df = df.groupby("release_year")["vote_average"].mean()
fig= sns.lineplot(x = average_df.index, y = average_df.values)
plt.xticks(np.arange(df["release_year"].min(), df["release_year"].max()+1, 10))
plt.suptitle("Average Vote_average per Year", fontsize = 20, fontweight = "bold")
plt.ylabel("Average Vote Average", fontweight = "bold")
plt.xlabel("Release Year", fontweight = "bold")
plt.show()

#Creates a bar plot of genre frequency
all_genres = df['genres'].dropna().str.split(',').explode().str.strip()
genre_counts = Counter(all_genres).most_common()
genre_df = pd.DataFrame(genre_counts, columns=['genre', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=genre_df, x='count', y='genre', color='steelblue')
plt.title('Genre Frequency — Which Genres Dominate the Dataset?', fontsize=14, fontweight='bold')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

#TEXT VECTORIZATION ______________________________________
#Count Vectorizer
def count_vectorization (columns):
    """
    A function to vectorize a standardized text column using CountVectorizer()

    Args:
        columns: pandas.Series. A pandas column with standardized text data
    Return:
        count_matrix: scipy sparse matrix. A sparse matrix with vectorized data
    """

    #Initialize the vectorization model
    count_vectorizer = CountVectorizer()

    #Transform the column into vectorized data
    count_matrix = count_vectorizer.fit_transform(df[columns])

    #Return the column
    return count_matrix

#Apply count vectorizer onto the "columns" column
count_matrix = count_vectorization("columns")

#TFIDF Vectorizer
def tfidf_vectorization (df, columns):
  '''
  This function turns a text column into a tfidf matrix
  Args:
    df (pandas data frame): Dataframe containing text column
    column (string): Name of the column in the pandas data frame to vectorize
  Return:
    matrix (tfidf matrix): tfidf matrix of the text column
    features: (np array): array of essential words learned
  '''
  tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000)
  matrix = tfidf.fit_transform(df[columns])
  #Returns a numpy array of essential words learned
  features = tfidf.get_feature_names_out()

  return matrix, features
    
#Runs TFIDF on our selected columns
tfidf_matrix, features = tfidf_vectorization(df, "columns")

#Turns the matrix into a dense Numpy Array
array_tfidf_matrix = tfidf_matrix.toarray()

#creates a new data frame of the first 10 films in the TFIDF Matrix
heat_map_df = pd.DataFrame(array_tfidf_matrix[:10], columns = features, index = df["title"][:10])

#Organizes the dataframe so that we select the 20 most influential words
top_words = heat_map_df.sum().sort_values(ascending = False).head(20).index

#Creates a HeatMap of TFIDF scores
plt.figure(figsize=(12,6))
sns.heatmap(heat_map_df[top_words], cmap ="RdPu")
plt.title("TF-IDF Vectorization", fontsize = 20, fontweight = "bold")
plt.xlabel("Features", fontweight = "bold")
plt.ylabel("Movies", fontweight = "bold")
plt.show()

# Word2Vec Vectorization
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

sentences = [row.split() for row in df['columns']]

# Train the Word2Vec model
# vector_size: dimensionality of the word vectors
# window: maximum distance between the current and predicted word within a sentence
# min_count: ignores all words with total frequency lower than this
# workers: use this many worker threads to train the model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=15)

def get_document_vector(text_tokens, model, vector_size):
    vectors = []
    for word in text_tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    # dedented — runs after the full loop
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

    # Create a new column in the DataFrame for the Word2Vec document vectors
df['word2vec_vectors'] = [get_document_vector(tokens, word2vec_model, 100) for tokens in sentences]

#Put all vectorized rows into a numpy array
word2vec_vector = np.array([get_document_vector(tokens, word2vec_model, 100) for tokens in sentences])

#Dimension reduction
def reduce_dimension(vectorized_column, n_comp):
    """
    A function to reduce the dimensions of a vectorized array or matrix into n dimensions using PCA.

    Args:
        vectorized_column: numpy array or scipy sparse matrix. A set of vectorized data
        n_comp: int. The number of dimensions you wish to reduce the array/matrix to
    Return:
        reduced_vector: numpy array. A numpy array containing the reduced dimensions of each row
    """
    #Initialize the PCA model to reduce the vector to n_comp dimensions
    pca = PCA(n_components = n_comp)

    #Reducing the original vectorized array
    reduced_vector = pca.fit_transform(vectorized_column)

    #Calculating the cumulative expalined variance ratio of all dimensions
    cumulative_ratio = np.cumsum(pca.explained_variance_ratio_)

    #Print the cumulative explained variance ratio
    print("Cumulative Explained Variance Ratio: " + str(cumulative_ratio[n_comp - 1]))

    #Return the reduced array
    return reduced_vector

#Reduce count_matrix to 100 features
count_matrix_reduced = reduce_dimension(count_matrix, 100)

#Reduce tfidf_matrix to 100 features
tfidf_matrix_reduced = reduce_dimension(tfidf_matrix, 100)

# MACHINE LEARNING CLUSTERING _______________________________________
# you must pip install scikit-fuzzy 
def find_best_clusters(matrix, min_clusters=2, max_clusters=10):
  '''
  This function finds the best number of clusters for a fuzzy c-means clustering algorithm.
  Args:
    data (vectorized matrix): vectorized matrix of text
    min_clusters (int): the minimum number of clusters to test
    max_clusters (int): the maximum number of clusters to test
  Return:
    best_clusters (int): the best number of clusters
    best_fpc (float): the best fuzzy partition coefficient
  '''
  fpc_values = [] #records all fpc_values for each n_cluster
  cluster_range = range(min_clusters, max_clusters + 1) #possible n_clusters

  for n_clusters in cluster_range:
      _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(matrix, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
      fpc_values.append(fpc)

  # Best cluster count
  best_index = np.argmax(fpc_values) #returns index of largest value
  best_clusters = list(cluster_range)[best_index] #returns the best # of clusters
  best_fpc = fpc_values[best_index] #returns the highest fpc score

  # Plots the process
  plt.plot(cluster_range, fpc_values)
  plt.xlabel("Number of Clusters")
  plt.ylabel("FPC")
  plt.title("Fuzzy Partition Coefficient")

  return best_clusters, best_fpc
    
def fuzzy_clustering(matrix, best_clusters, fuzziness_param = 2, error = 1e-5, maxiter =1000):
    """
    Fuzzy C-means clustering algorithm
    Args:
        Matrix: Reduced Vectorized Matrix
        best_cluster: Best number of clusters
        fuzziness_param: Fuzziness parameter
        maxiter: max number of iterations
    Returns:
        Cluster membership of each movie
    """
    #transpose Matrix
    matrix = matrix.T
    #Fuzzy C-means Clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(matrix, best_clusters, fuzziness_param, error, maxiter)
    #Determines cluster membership of each movie
    cluster_membership = np.argmax(u, axis = 0)
    return cluster_membership

def run_gmm(vectorized_column, n_components=8):
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(vectorized_column)
    membership = gmm.predict(vectorized_column)

    print("Cluster distribution:")
    print(df["gmm_cluster"].value_counts().sort_index())

    # Visualization
    plot_df = pd.DataFrame({
        "Movie Title":      df["title"],
        "First Dimension":  vectorized_column[:, 0],
        "Second Dimension": vectorized_column[:, 1],
        "Cluster":          df["gmm_cluster"].astype(str)
    })

    fig = px.scatter(
        plot_df,
        x="First Dimension",
        y="Second Dimension",
        color="Cluster",
        hover_data="Movie Title",
        title="Distribution of Genres, Keywords, and Overview Using Word2Vec"
    )
    fig.show()

    return membership

# RECOMMENDATION ALGORITHMS _______________________________________
#Find Cluster
def find_from_cluster(movie, class_type, vectorized_column):
    """
    A helper function that find the cluster that the movie belongs to and find all movies from that cluster in the vectorized array.

    Args:
        movie: string. The title of the movie you are trying to find the cluster of
        class_type: string. The cluster model you are trying to find the movie in
        vectorized_column: numpy array. The reduced vectorized array that you wish to create a subset of
    Return:
        movie_index: pd.Index. The index of the movie based on its position in the subsetted dataframe
        subset_of_vector: numpy array. A subset of the original vectorized array containing only the movies from the cluster the inputted
            movie belongs in
    """
    #Find the class of the movie
    class_index = df[df["title"] == movie][class_type]

    #If there are multiple movies with the same title, take the first movie
    if class_index.shape != (1,):
        class_index = int(class_index.iloc[0])
    else:
        class_index = int(class_index)

    #Create a subset with all movies from the same cluster as the input
    subset = df[df[class_type] == class_index].reset_index(drop = True)

    #Find the index of the movie from the subset
    movie_index = subset[subset["title"] == movie].index

    #Create a subset with the corresponding vectorized data for each movie from the given cluster
    subset_of_vector = vectorized_column[subset.index]

    #Return a list with the new movie index and the subset of vectorzed data
    return [movie_index, subset_of_vector]

#Cosine Similarity
def similarity_of_movies(movie, vectorized_column, cluster_model = None):
    """
    A function to find recommendations for a movie using the cosine similarity metric. This version is for repeated experiments.

    Args:
        movie: string. The title of movie you wish to find a recommendation for
        vectorized_column: numpy array. The vectorization technique you wish to use for your recommendation
        cluster_model: string (default = None). The name of the column that corresponds to the vectorization technique you are inputting
            and the clustering model you wish to use
    Return:
        similarity_score: float. The similarity score between the inputted movie and the recommended movie
        recommended_title: pandas dataframe. The dataframe of the recommended movie and all of its features
    """
    #For when we are passing in a cluster
    if cluster_model != None:
        #Call the helper function to find the new movie index and the subset of the vectorized array 
        #based on the cluster that the movie belongs to
        movie_index, vectorized_column = find_from_cluster(movie, cluster_model, vectorized_column)
    else:
        #Find the index of the movie in the dataframe
        movie_index = df[df["title"] == movie].index

    #Compute the similarity between movies in the dataframe
    similarity = cosine_similarity(vectorized_column[movie_index], vectorized_column)

    #Sort all of the movies' indices by similarity scores, from highest to lowest, and take the second index (so the recommendation
    #isn't the input)
    index = np.argsort(-similarity)[0, 1]

    #If the second similarity score isn't the input, take the index of the first score
    if (df["title"].iloc[index] == movie):
        index = np.argsort(-similarity)[0, 0]

    #Subset the recommended movie from the dataset
    recommended_title = df.iloc[index]

    #Find the similarity score of the recommended movie
    similarity_score = similarity[0][index]

    #Report cosine similarity scores
    print(df["title"].iloc[index] + ": " + str(similarity[0][index]))

    #Return the similarity score and a dataframe with the most similar movie
    return (similarity_score, recommended_title)

#Nearest Neighbors
def nearest_neighbors(movie, vectorized_column, cluster_model = None):
    """
    A function to find recommendations for a movie using nearest neighbors. This version is for repeated experiments.

    Args:
        movie: string. The title of movie you wish to find a recommendation for
        vectorized_column: numpy array. The vectorization technique you wish to use for your recommendation
        cluster_model: string (default = None). The name of the column that corresponds to the vectorization technique you are inputting
            and the clustering model you wish to use
    Return:
        distance_score: float. The distance score between the inputted movie and the recommended movie
        recommended_title: pandas dataframe. The dataframe of the recommended movie and all of its features 
    """
    #For when we are passing in a cluster
    if cluster_model != None:
        #Call the helper function to find the subset based on the cluster that the movie belongs to
        movie_index, vectorized_column = find_from_cluster(movie, cluster_model, vectorized_column)
    else:
        #Find the index of the movie in the dataframe
        movie_index = df[df["title"] == movie].index
    
    #Initialize the model
    nbrs = NearestNeighbors(n_neighbors = 2, algorithm = "ball_tree")

    #Train the model
    nbrs.fit(vectorized_column)

    #Gather the indices and distances of the nearest neighbor
    (distance, index) = nbrs.kneighbors(vectorized_column[movie_index])

    #Flatten the distance array
    distance = distance.flatten()

    #Flatten the index array and make sure to remove the inputted movie
    index = index.flatten()
    index = index[np.isin(index, movie_index, invert = True)]

    #For when there is more than one index, take the first index
    if (index.shape != (1,)):
        index = index[1]

    #Convert the index to int for easier index
    index = int(index)
    
    #Store the recommenced title without returning the inputted movie itself
    recommended_title = df.iloc[index]

    #Store the distance between the inputted movie and the recommended movie
    distance_score = distance[1]
    
    #Report distances between the inputted movie and the recommended movie
    print(df["title"].iloc[index] + ": " + str(distance[1]))

    #Return the distance score and the recommended movie
    return (distance_score, recommended_title)

#Final recommendation algorithm
def recommend(movie, vectorized_column, cluster_model = None):
    """
    A function to find recommendations for a movie using the cosine similarity metric. This version is for performing recommendations.

    Args:
        movie: string. The title of movie you wish to find a recommendation for
        vectorized_column: numpy array. The vectorization technique you wish to use for your recommendation
        cluster_model: string (default = None). The name of the column that corresponds to the vectorization technique you are inputting
            and the clustering model you wish to use
    Return:
        recommended_title: pandas dataframe. A dataframe containing three recommended movies
    """
    #For when we are passing in a cluster
    if cluster_model != None:
        #Call the helper function to find the new movie index and the subset of the vectorized array 
        #based on the cluster that the movie belongs to
        movie_index, vectorized_column = find_from_cluster(movie, cluster_model, vectorized_column)
    else:
        #Find the index of the movie in the dataframe
        movie_index = df[df["title"] == movie].index

    #Compute the similarity between movies in the dataframe
    similarity = cosine_similarity(vectorized_column[movie_index], vectorized_column)

    #Sort all of the movies' indices by similarity scores, from highest to lowest, and take the second index (so the recommendation
    #isn't the input)
    index = np.argsort(-similarity)[0, 1:4]

    #Subset the recommended movies from the dataset
    recommended_titles = df.iloc[index].reset_index(drop = True)
    recommended_titles = recommended_titles[["title", "release_date", "overview", "genres"]]

    #Return the similarity score and a dataframe with the most similar movie
    return recommended_titles
