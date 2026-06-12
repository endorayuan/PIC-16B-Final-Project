#Importing all necessary libraries
import pandas as pd
import numpy as np
from numpy import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import sys
import warnings
import os

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score



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
    A function to find recommendations for a movie using the cosine similarity metric. 

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
    A function to find recommendations for a movie using nearest neighbors.

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
