import myProject

def repeated_experiments(metric, movie_list, vector, cluster = None):
    """
    A function to test the different combinations of similarity metrics, vectorization techniques, and clustering models.

    Args:
        metric: function. The similarity metric you wish to test
        movie_list: pandas series. A list of movies
        vector: numpy array. The vectorized array that you wish to test
        cluster: string (default = None). The clustering model you wish to test
    Return:
        ave_score: float. The average score of the combination, computed across the length of movie_list
    """
    #Initialize an empty list to store the score of each run
    ave_score = []

    #Run the combination of metric, vectorization technique, and clustering model and record the score
    for i in movie_list:
        ave_score.append(metric(i, vector, cluster)[0])

    #Conver ave_score to an array and compute the mean of all runs
    ave_score = np.array(ave_score)
    ave_score = np.mean(ave_score)

    #Print the average score across all runs for the combination
    print('')
    print("Average Score: " + str(ave_score))

    #Return the average score of all runs for the combination 
    return ave_score

#Testing process
#Create a pandas series of 10 movies, randomly selected from the dataframe
random_movies = random.randint(204389, size = (10))
random_movies = df["title"].iloc[random_movies]
random_movies

#Initialize a list to store all of the cosine similarity scores
cosine_scores = []

#Initialize a list to store all of the nearest neighbors score
neighbor_scores = []

#Cosine similarity only and count vectorizer
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          count_matrix_reduced))

#Cosine similarity only and TFIDF matrix
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          tfidf_matrix_reduced))

#Cosine similarity only and Word2Vec
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          word2vec_vector))

#Cosine similarity, fuzzy clustering, and count vectorizer
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          count_matrix_reduced, 
                                          "fuzzy count"))

#Cosine similarity, fuzzy clustering, and TFIDF
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          tfidf_matrix_reduced, 
                                          "fuzzy tfidf"))

#Cosine similarity, fuzzy clustering, and Word2Vec
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          word2vec_vector, 
                                          "fuzzy word2vec"))

#Cosine similarity, GMM, and count vectorizer
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          count_matrix_reduced, 
                                          "gmm count"))

#Cosine similarity, GMM, and TFIDF
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          tfidf_matrix_reduced, 
                                          "gmm tfidf"))
#Cosine similarity, GMM, and Word2Vec
cosine_scores.append(repeated_experiments(similarity_of_movies, 
                                          random_movies, 
                                          word2vec_vector, 
                                          "gmm word2vec"))

#Nearest neighbors only and count vectorizer
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            count_matrix_reduced))

#Nearest neighbors only and TFIDF
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            tfidf_matrix_reduced))

#Nearest neighbors only and Word2Vec
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            word2vec_vector))

#Nearest neighbors, fuzzy clustering, and count vectorizer
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            count_matrix_reduced, 
                                            "fuzzy count"))

#Nearest neighbors, fuzzy clustering, and TFIDF
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            tfidf_matrix_reduced, 
                                            "fuzzy tfidf"))

#Nearest neighbors, fuzzy clustering, and Word2Vec
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            word2vec_vector, 
                                            "fuzzy word2vec"))

#Nearest neighbors, GMM, and count vectorizer
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            count_matrix_reduced, 
                                            "gmm count"))

#Nearest neighbors, GMM, and TFIDF
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            tfidf_matrix_reduced, 
                                            "gmm tfidf"))

#Nearest neighbors, GMM, and Word2Vec
neighbor_scores.append(repeated_experiments(nearest_neighbors, 
                                            random_movies, 
                                            word2vec_vector, 
                                            "gmm word2vec"))

#Create a dictionary to map the combination to their similarity scores
cosine_similarity_scores = {"Vectorization Techniques": ["Count Vectorizer", "TFIDF", "Word2Vec"],
                            "Cosine Similarity" : [cosine_scores[0], cosine_scores[1], cosine_scores[2]], 
                            "Fuzzy Clustering + Cosine Similarity" : [cosine_scores[3], cosine_scores[4], cosine_scores[5]],
                            "GMM + Cosine Similarity" : [cosine_scores[6], cosine_scores[7], cosine_scores[8]]}

#Create a dictionary to map the combination to their distance scores
nearest_neighbors_scores = {"Vectorization Technique": ["Count Vectorizer", "TFIDF", "Word2Vec"],
                            "Nearest Neighbors" : [neighbor_scores[0], neighbor_scores[1], neighbor_scores[2]],
                            "Fuzzy Clustering + Nearest Neighbors" : [neighbor_scores[3], neighbor_scores[4], neighbor_scores[5]],
                            "GMM + Nearest Neighbors" : [neighbor_scores[6], neighbor_scores[7], neighbor_scores[8]]}

#Create a dataframe to display all of the average similarity scores
cosine_similarity = pd.DataFrame(data = cosine_similarity_scores)
pd.set_option("display.precision", 4)

#Create a dataframe to display all of the average distance scores
nearest_neighbors = pd.DataFrame(data = nearest_neighbors_scores)




