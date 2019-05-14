from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from itertools import chain
from collections import Counter
from pandas.io.json import json_normalize

import sklearn.metrics.pairwise as pw 
import numpy as np
import pandas as pd
import random


def reviewed_businesses(user_id):
    """Returns a list of the businesses the current user left a review on """
    user_reviews =[]
    for reviews in REVIEWS.values():
        for review in reviews:
            if review['user_id'] == user_id:
                user_reviews.append(review['business_id'])
    return(user_reviews)

def user_categories(user_businesses):
    """ Return the categories of the restaurants where the user left a review
    Do this with a dict so that the most reviewed type business has a higher number.
    A list of businesses that the user reviewed on needs to be given
    """
    user_categories =[]
    for businesses in BUSINESSES.values():
        for business in businesses:
            if business['business_id'] in user_businesses:
                categories = business["categories"].split(", ")
                user_categories.append(categories)
    
    return Counter(chain.from_iterable(user_categories))

def categories_dataframe():
    """ Make a dataframe with the business_id's and their categories"""
    all_data = list()
    for businesses in BUSINESSES.values():
        for business in businesses:
            business_id = business['business_id']
            categories = business['categories']
    
    # add to the data collected so far
            all_data.append([business_id, categories])

    # create the DataFrame
    categories_df = pd.DataFrame(all_data, columns=['business_id', 'categories'])
    return categories_df

def extract_categories():
    """" extract the categories"""
    businesses = categories_dataframe()
    categories_b = businesses.apply(lambda row: pd.Series([row['business_id']] + row['categories'].split(", ")), axis=1)
    stack_categories = categories_b.set_index(0).stack()
    df_stack_categories = stack_categories.to_frame()
    df_stack_categories['business_id'] = stack_categories.index.droplevel(1)
    df_stack_categories.columns = ['categories', 'business_id']
    return df_stack_categories.reset_index()[['business_id', 'categories']]

def utility_categories(df):
    """ Returns the utility matrix for all the businesses and their categories"""
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def business_categories(dataframe, cat_dict):
    """ Return a dataframe of businesses that fit the categories of the user."""
    categories = list(cat_dict.keys())
    df_new = dataframe[categories]
    return df_new[(df_new.T != 0).any()]

def business_rating(user_id, user_businesses):
    """ Return a dict with the ratings that users gave for a list of businesses"""
    user_ratings = {}
    for reviews in REVIEWS.values():
        for review in reviews:
            if review['business_id'] in user_businesses and review['user_id'] == user_id:
                user_ratings[review['business_id']] = review['stars']
    return user_ratings



##################################################
#            """ CONTENT BASED """"
##################################################

def split_data(data, d = 0.75):
    """Split data in a training and test set.
    
    Arguments:
    data -- any dataFrame.
    d    -- the fraction of data in the training set
    """
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]

def json_to_df_stars():
    """Converts all review.jsons to a single DataFrame containing the columns business_id and user_id"""
    df = pd.DataFrame()

    # add each city's DataFrame to the general DataFrame
    for city in CITIES:
        reviews = REVIEWS[city]
        df = df.append(pd.DataFrame.from_dict(json_normalize(reviews), orient='columns'))
    
    # drop repeated user/business reviews and only save the latest one (since that one is most relevant)
    df = df.drop_duplicates(subset=["business_id", "user_id"], keep="last").reset_index()[["business_id", "stars", "user_id"]]
    return df

def extract_categories():
    """" extract the categories"""
    businesses = categories_dataframe()
    categories_b = businesses.apply(lambda row: pd.Series([row['business_id']] + row['categories'].split(", ")), axis=1)
    stack_categories = categories_b.set_index(0).stack()
    df_stack_categories = stack_categories.to_frame()
    df_stack_categories['business_id'] = stack_categories.index.droplevel(1)
    df_stack_categories.columns = ['categories', 'business_id']
    return df_stack_categories.reset_index()[['business_id', 'categories']]


def create_similarity_matrix_cosine(matrix):
    """Creates a adjusted(/soft) cosine similarity matrix.
    
    Arguments:
    matrix -- a utility matrix
    
    Notes:
    Missing values are set to 0. This is technically not a 100% correct, but is more convenient 
    for computation and does not have a big effect on the outcome.
    """
    mc_matrix = matrix - matrix.mean(axis = 0)
    return pd.DataFrame(pw.cosine_similarity(mc_matrix.fillna(0)), index = matrix.index, columns = matrix.index)

def pivot_ratings(df):
    """Creates a utility matrix for user ratings for movies
    
    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genres'
    
    Output:
    a matrix containing a rating in each cell. np.nan means that the user did not rate the movie
    """
    return df.pivot(values='stars', columns='user_id', index='business_id')


def predict_ratings(similarity, utility, to_predict):
    """Predicts the predicted rating for the input test data.
    
    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each movie (rows). 
                  If a user did not rate an item the value np.nan is assumed. 
    to_predict -- A dataFrame containing at least the columns movieId and userId for which to do the predictions
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id']), axis=1)
    return ratings_test_c

### Helper functions for predict_ratings_item_based ###

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return np.nan

def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return np.nan
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm

def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()

def extract_genres(dataframe):
    """Create an unfolded genre dataframe. Unpacks genres seprated by a '|' into seperate rows.

    Arguments:
    movies -- a dataFrame containing at least the columns 'movieId' and 'genres' 
              where genres are seprated by '|'
    """
    genres_m = dataframe.apply(lambda row: pd.Series([row['business_id']] + row['categories'].lower().split(", ")), axis=1)
    stack_genres = genres_m.set_index(0).stack()
    df_stack_genres = stack_genres.to_frame()
    df_stack_genres['business_id'] = stack_genres.index.droplevel(1)
    df_stack_genres.columns = ['categories', 'business_id']
    return df_stack_genres.reset_index()[['business_id', 'categories']]

def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genre'
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)


# make a training and test set
df = json_to_df_stars()
df_training, df_test = split_data(df, d = 0.9)
# make the utility matrix with the amount of stars
df_utility_stars = pivot_ratings(df_training)

# create the dataframe with the business id's and categories
categories_dataframe = extract_categories()

# make utility matrix with the categories
df_utility_categories = pivot_genres(categories_dataframe)

# make a similarity matrix with the use of the categories utility matrix
df_similarity_categories = create_similarity_matrix_categories(df_utility_categories)

# predict the ratings
predicted_ratings = predict_ratings(df_similarity_categories, df_utility_stars, df_test)
predicted_ratings.dropna()
print(predicted_ratings)

# calculate the mse for content based filtering
mse_content = mse(predicted_ratings)
print(mse_content)